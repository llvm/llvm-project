//===-- SwiftExpressionSourceCode.cpp --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SwiftExpressionSourceCode.h"
#include "Plugins/LanguageRuntime/Swift/SwiftLanguageRuntime.h"
#include "SwiftPersistentExpressionState.h"

#include "Plugins/ExpressionParser/Swift/SwiftASTManipulator.h"
#include "Plugins/TypeSystem/Swift/SwiftASTContext.h"
#include "lldb/Symbol/Variable.h"
#include "lldb/Target/Language.h"
#include "lldb/Target/Platform.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/StreamString.h"

#include "swift/Basic/LangOptions.h"
#include "swift/Demangling/Demangle.h"
#include "swift/Demangling/Demangler.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/Dwarf.h"

using namespace lldb;
using namespace lldb_private;

namespace lldb_private {
std::optional<std::pair<unsigned, unsigned>>
ParseSwiftGenericParameter(llvm::StringRef name) {
  if (!name.consume_front("$τ_"))
    return {};

  auto pair = name.split('_');
  auto depth_str = pair.first;
  auto index_str = pair.second;
  unsigned depth, index;
  if (depth_str.getAsInteger(10, depth) || index_str.getAsInteger(10, index))
    return {};
  return {{depth, index}};
}
} // namespace lldb_private

static const char *GetUserCodeStartMarker() {
  return "/*__LLDB_USER_START__*/\n";
}
static const char *GetUserCodeEndMarker() { return "\n/*__LLDB_USER_END__*/"; }

/// Remove SILPacktype and print the name with substitutions applied.
static llvm::Expected<std::string> TransformPackType(
    CompilerType type,
    llvm::SmallDenseMap<std::pair<unsigned, unsigned>, llvm::SmallString<4>>
        subs) {
  auto tss = type.GetTypeSystem().dyn_cast_or_null<TypeSystemSwift>();
  if (!tss)
    return llvm::createStringError(llvm::errc::not_supported,
                                   "unexpected typesystem");
  auto ts = tss->GetTypeSystemSwiftTypeRef();
  if (!ts)
    return llvm::createStringError(llvm::errc::not_supported,
                                   "no typeref typesystem");

  auto mangled_name = type.GetMangledTypeName().GetStringRef();
  auto flavor = SwiftLanguageRuntime::GetManglingFlavor(mangled_name);

  using namespace swift::Demangle;
  Demangler dem;

  NodePointer node = ts->GetCanonicalDemangleTree(dem, mangled_name);

  node = TypeSystemSwiftTypeRef::Transform(dem, node, [](NodePointer n) {
    if (n->getKind() == Node::Kind::SILPackIndirect &&
        n->getNumChildren() == 1) {
      n = n->getFirstChild();
      if (n->getKind() == Node::Kind::Type && n->getNumChildren() == 1)
        return n->getFirstChild();
    }
    return n;
  });

  bool error = false;
  ConstString type_name =
      ts->RemangleAsType(dem, node, flavor).GetMangledTypeName();
  swift::Demangle::DemangleOptions options;
  options = swift::Demangle::DemangleOptions::SimplifiedUIDemangleOptions();
  options.DisplayStdlibModule = false;
  options.DisplayObjCModule = false;
  options.QualifyEntities = true;
  options.DisplayModuleNames = true;
  options.DisplayLocalNameContexts = false;
  options.DisplayDebuggerGeneratedModule = false;
  options.GenericParameterName = [&](uint64_t depth,
                                     uint64_t index) -> std::string {
    auto it = subs.find({depth, index});
    if (it != subs.end())
      return it->second.str().str();
    error = true;
    return "$error";
  };
  std::string name = swift::Demangle::demangleSymbolAsString(
      type_name.GetStringRef(), options);
  if (error)
    return llvm::createStringError(llvm::errc::not_supported,
                                   "unexpected generic parameter");
  return name;
}

struct CallsAndArgs {
  std::string lldb_user_expr;
  std::string lldb_trampoline;
  std::string lldb_sink;
  std::string lldb_call;
};

static llvm::SmallVector<const SwiftASTManipulator::VariableInfo *>
CollectMetadataVariables(
    llvm::ArrayRef<SwiftASTManipulator::VariableInfo> local_variables) {
  llvm::SmallVector<const SwiftASTManipulator::VariableInfo *>
      metadata_variables;
  for (auto &var : local_variables)
    if (var.IsOutermostMetadataPointer())
      metadata_variables.push_back(&var);
  return metadata_variables;
}

struct MetadataInfo {
  bool is_pack = false;
  unsigned depth;
  unsigned index;
  llvm::StringRef archetype_name;

  MetadataInfo(bool is_pack, unsigned depth, unsigned index, llvm::StringRef archetype_name)
      : is_pack(is_pack), depth(depth), index(index), archetype_name(archetype_name) {}
};

static llvm::Expected<llvm::SmallVector<MetadataInfo>> CollectMetadataInfos(
    llvm::ArrayRef<const SwiftASTManipulator::VariableInfo *>
        metadata_variables,
    const std::optional<SwiftLanguageRuntime::GenericSignature> &generic_sig) {
  llvm::SmallVector<MetadataInfo> metadata_info;

  for (size_t i = 0; i < metadata_variables.size(); ++i) {
    auto variable_sp =
        llvm::cast<SwiftASTManipulatorBase::VariableMetadataVariable>(
            metadata_variables[i]->GetMetadata())
            ->m_variable_sp;
    auto archetype_name = variable_sp->GetType()->GetName();
    if (generic_sig)
      metadata_info.emplace_back(generic_sig->generic_params[i].is_pack,
                                 generic_sig->generic_params[i].depth,
                                 generic_sig->generic_params[i].index,
                                 archetype_name);
    else {
      auto maybe_depth_and_index =
          ParseSwiftGenericParameter(metadata_variables[i]->GetName().str());
      if (!maybe_depth_and_index)
        return llvm::createStringError(llvm::errc::not_supported,
                                       "unexpected metadata variable");
      metadata_info.emplace_back(false, maybe_depth_and_index->first,
                                 maybe_depth_and_index->second, archetype_name);
    }
  }
  return metadata_info;
}
/// Constructs the signatures for the expression evaluation functions based on
/// the metadata variables in scope and any variadic functiontion parameters.
/// For every outermost metadata pointer in scope ($τ_0_0, $τ_0_1, etc), we want
/// to generate:
///
/// - A $__lldb_user_expr signature that takes in that many metadata pointers:
///
/// func $__lldb_user_expr<T0, T1, ..., Tn>
///     (_ $__lldb_arg: UnsafeMutablePointer<(T0, T1, ..., Tn)>)
///
/// - An optional $__lldb_trampoline signature like the above, but
///   that also takes in a pointer to self:
///
/// func $__lldb_trampoline<T0, T1, ..., Tn>
///      (_ $__lldb_arg: UnsafeMutablePointer<(T0, T1, ..., Tn)>,
///       _ $__lldb_injected_self: inout $__lldb_context)
///
/// - A $__lldb_sink signature that matches the number of parameters of the
///   trampoline:
///
/// func $__lldb_sink(_ $__lldb_arg : UnsafeMutablePointer<Any>,
///                   _: $__lldb_builtin_ptr_t, // the self variable
///                   _: $__lldb_builtin_ptr_t, // T0
///                   _: $__lldb_builtin_ptr_t, // T1
///                   ...,
///                   _: $__lldb_builtin_ptr_t) // Tn
///
/// - And a matching call to the sink function:
///
/// lldb_sink($__lldb_arg, [$__lldb_injected_self, [pack args, pack counts...]],
///           $τ_0_0, $τ_0_1, ..., $τ_0_n)
static llvm::Expected<CallsAndArgs> MakeGenericSignaturesAndCalls(
    llvm::ArrayRef<SwiftASTManipulator::VariableInfo> local_variables,
    const std::optional<SwiftLanguageRuntime::GenericSignature> &generic_sig,
    bool needs_object_ptr) {
  auto metadata_variables = CollectMetadataVariables(local_variables);
  // The number of metadata variables could be > if the function is in
  // a generic context.
  if (generic_sig &&
      (metadata_variables.size() < generic_sig->dependent_generic_param_count))
    return llvm::createStringError(llvm::errc::not_supported,
                                   "Inconsistent generic signature");

  auto maybe_metadata_infos = CollectMetadataInfos(metadata_variables, generic_sig);
  if (!maybe_metadata_infos)
    return maybe_metadata_infos.takeError();
  auto metadata_infos = *maybe_metadata_infos;

  llvm::SmallDenseMap<std::pair<unsigned, unsigned>, llvm::SmallString<4>> subs;
  std::string generic_params;
  std::string generic_params_no_packs;
  llvm::raw_string_ostream s_generic_params(generic_params);
  llvm::raw_string_ostream s_generic_params_no_packs(generic_params_no_packs);

  for (auto &[is_pack, depth, index, archetype_name] : metadata_infos) {
    llvm::SmallString<4> sig_archetype_name;
    llvm::raw_svector_ostream s_sig_archetype_name(sig_archetype_name);
    if (is_pack)
      s_sig_archetype_name << "each ";
    s_sig_archetype_name << archetype_name;
    if (!is_pack)
      s_generic_params_no_packs << sig_archetype_name << ",";
    s_generic_params << sig_archetype_name << ",";
    subs.insert({{depth, index}, sig_archetype_name});
  }

  if (!generic_params.empty())
    generic_params.pop_back();
  if (!generic_params_no_packs.empty())
    generic_params_no_packs.pop_back();

  std::string user_expr; 
  llvm::raw_string_ostream user_expr_stream(user_expr);
  user_expr_stream << "func $__lldb_user_expr<" << generic_params
                   << ">(_ $__lldb_arg: UnsafeMutablePointer<("
                   << generic_params_no_packs << ")>";
  for (auto &var : local_variables)
    if (var.GetType().GetTypeInfo() & lldb::eTypeIsPack) {
      auto pack_type = TransformPackType(var.GetType(), subs);
      if (!pack_type)
        return pack_type.takeError();
      user_expr_stream << ", _ " << var.GetName() << ": "
                       << *pack_type;
    }
  user_expr_stream << ")";

  std::string trampoline;
  llvm::raw_string_ostream trampoline_stream(trampoline);
  trampoline_stream << "func $__lldb_trampoline<" << generic_params
                    << ">(_ $__lldb_arg: UnsafeMutablePointer<("
                    << generic_params_no_packs << ")>";
  if (needs_object_ptr)
    trampoline_stream << ", _ $__lldb_injected_self: inout $__lldb_context";
  trampoline_stream << ")";

  std::string sink;
  std::string call;
  llvm::raw_string_ostream sink_stream(sink);
  llvm::raw_string_ostream call_stream(call);
  sink_stream << "func $__lldb_sink(_ $__lldb_arg : "
    "UnsafeMutablePointer<Any>";
  call_stream << "$__lldb_sink($__lldb_arg";
  if (needs_object_ptr) {
    sink_stream << ", _: $__lldb_builtin_ptr_t";
    call_stream << ", $__lldb_injected_self";
  }
  unsigned num_value_packs = 0;
  for (auto &var : local_variables)
    if (var.IsUnboundPack()) {
      ++num_value_packs;
      sink_stream << ", _: $__lldb_builtin_ptr_t";
      call_stream << ", " << var.GetName();
    }
  // FIXME: This assumes all pack variables are local function arguments.
  assert(!generic_sig ||
         num_value_packs == generic_sig->pack_expansions.size());

  if (generic_sig)
    for (unsigned i = 0; i < generic_sig->num_counts; ++i) {
      sink_stream << ", _: $__lldb_builtin_int_t";
      call_stream << ", $pack_count_" << i;
    }
  for (auto &var : metadata_variables) {
    sink_stream << ", _: $__lldb_builtin_ptr_t";
    call_stream << ", " << var->GetName().str();
  }
  sink_stream << ")";
  call_stream << ")";

  CallsAndArgs retval = {user_expr, trampoline, sink, call};
  return retval;
}

static Status WrapExpression(
    lldb_private::Stream &wrapped_stream, const char *orig_text,
    bool needs_object_ptr, bool static_method, bool is_class, bool weak_self,
    const EvaluateExpressionOptions &options, llvm::StringRef os_version,
    uint32_t &first_body_line,
    llvm::ArrayRef<SwiftASTManipulator::VariableInfo> local_variables,
    const std::optional<SwiftLanguageRuntime::GenericSignature> &generic_sig) {
  Status status;
  first_body_line = 0; // set to invalid
  // TODO make the extension private so we're not polluting the class
  static unsigned int counter = 0;
  unsigned int current_counter = counter++;

  const bool playground = options.GetPlaygroundTransformEnabled();
  const bool repl = options.GetREPLEnabled();
  const bool generate_debug_info = options.GetGenerateDebugInfo();
  const char *pound_file = options.GetPoundLineFilePath();
  const uint32_t pound_line = options.GetPoundLineLine();
  const char *text = orig_text;
  StreamString fixed_text;

  if (playground) {
    const char *playground_logger_declarations = R"(
@_silgen_name ("playground_logger_initialize") func __builtin_logger_initialize ()
@_silgen_name ("playground_log_hidden") func __builtin_log_with_id<T> (_ object : T, _ name : String, _ id : Int, _ sl : Int, _ el : Int, _ sc : Int, _ ec: Int, _ moduleID: Int, _ fileID: Int) -> AnyObject
@_silgen_name ("playground_log_scope_entry") func __builtin_log_scope_entry (_ sl : Int, _ el : Int, _ sc : Int, _ ec: Int, _ moduleID: Int, _ fileID: Int) -> AnyObject
@_silgen_name ("playground_log_scope_exit") func __builtin_log_scope_exit (_ sl : Int, _ el : Int, _ sc : Int, _ ec: Int, _ moduleID: Int, _ fileID: Int) -> AnyObject
@_silgen_name ("playground_log_postprint") func __builtin_postPrint (_ sl : Int, _ el : Int, _ sc : Int, _ ec: Int, _ moduleID: Int, _ fileID: Int) -> AnyObject
@_silgen_name ("DVTSendPlaygroundLogData") func __builtin_send_data (_ :  AnyObject!)
__builtin_logger_initialize()
)";

    // The debug function declarations need only be declared once per session -
    // on the first REPL call.  This code assumes that the first call is the
    // first REPL call; don't call playground once then playground || repl
    // again
    bool first_expression = options.GetPreparePlaygroundStubFunctions();

    const char *playground_prefix =
        first_expression ? playground_logger_declarations : "";

    if (pound_file && pound_line) {
      wrapped_stream.Printf("%s#sourceLocation(file: \"%s\", line: %u)\n%s\n",
                            playground_prefix, pound_file, pound_line,
                            orig_text);
    } else {
      // In 2017+, xcode playgrounds send orig_text that starts with a module
      // loading prefix (not the above prefix), then a sourceLocation specifier
      // that indicates the page name, and then the page body text.  The
      // first_body_line mechanism in this function cannot be used to
      // compensate for the playground_prefix added here, since it incorrectly
      // continues to apply even after sourceLocation directives are read from
      // the orig_text.  To make sure playgrounds work correctly whether or not
      // they supply their own sourceLocation, create a dummy sourceLocation
      // here with a fake filename that starts counting the first line of
      // orig_text as line 1.
      wrapped_stream.Printf("%s#sourceLocation(file: \"%s\", line: %u)\n%s\n",
                            playground_prefix, "Playground.swift", 1,
                            orig_text);
    }
    first_body_line = 1;
    return status;
  }

  assert(!playground && "Playground mode not expected");

  if (repl) {
    if (pound_file && pound_line) {
      wrapped_stream.Printf("#sourceLocation(file: \"%s\", line:  %u)\n%s\n",
                            llvm::sys::path::filename(pound_file).str().c_str(),
                            pound_line, orig_text);
    } else {
      wrapped_stream.Printf("%s", orig_text);
    }
    first_body_line = 1;
    return status;
  }

  assert(!playground && !repl && "Playground/REPL mode not expected");

  auto path_literal = [](const char *path) -> std::string {
    std::string escaped;
    llvm::raw_string_ostream os(escaped);
    llvm::printEscapedString(path, os);
    return escaped;
  };

  if (pound_file && pound_line) {
    fixed_text.Printf("#sourceLocation(file: \"%s\", line: %u)\n%s\n",
                      path_literal(pound_file).c_str(), pound_line, orig_text);
    text = fixed_text.GetString().data();
  } else if (generate_debug_info) {
    std::string expr_source_path;
    if (SwiftASTManipulator::SaveExpressionTextToTempFile(orig_text, options,
                                                          expr_source_path)) {
      fixed_text.Printf("#sourceLocation(file: \"%s\", line: 1)\n%s\n",
                        path_literal(expr_source_path.c_str()).c_str(),
                        orig_text);
      text = fixed_text.GetString().data();
    }
  }

  // Note: All the wrapper functions we make are marked with the
  // @LLDBDebuggerFunction macro so that the compiler can do whatever special
  // treatment it need to do on them.  If you add new variants be sure to mark
  // them this way.  Also, any function that might end up being in an extension
  // of swift class needs to be marked final, since otherwise the compiler
  // might try to dispatch them dynamically, which it can't do correctly for
  // these functions.

  std::string availability = "";
  if (!os_version.empty())
    availability = (llvm::Twine("@available(") + os_version + ", *)").str();

  StreamString wrapped_expr_text;

  // Avoid indenting user code: this makes column information from compiler
  // errors match up with what the user typed.
  wrapped_expr_text.Printf(R"(
do {
%s%s%s
} catch (let __lldb_tmp_error) {
  var %s = __lldb_tmp_error
}
)",
                           GetUserCodeStartMarker(), text,
                           GetUserCodeEndMarker(),
                           SwiftASTManipulator::GetErrorName());

  if (needs_object_ptr || static_method) {
    const char *func_decorator = "";
    if (static_method) {
        func_decorator = "static";
    } else if (is_class && !weak_self) {
      func_decorator = "final";
    } else {
      func_decorator = "mutating";
    }

    const char *optional_extension =
        weak_self ? "Swift.Optional where Wrapped == " : "";

    // The expression text is inserted into the body of $__lldb_user_expr_%u.
    if (!SwiftASTManipulator::ShouldBindGenericTypes(
            options.GetBindGenericTypes())) {
      // A Swift program can't have types with non-bound generic type parameters
      // inside a non generic function. For example, the following program would
      // not compile as T is not part of foo's signature.
      //
      // func foo() {
      //   let bar: Baz<T> = ... // Where does T come from?
      //  }
      //
      // LLDB has to circumvent this problem, as the entry-point function for
      // expression evaluation can't be generic. LLDB achieves this by bypassing
      // the Swift typesystem, by:
      // - Setting the type of self in the expression to an opaque pointer type.
      // - Setting up $__lldb_trampoline, which calls the method
      // with the user's code. The purpose of this function is to have a common
      // number of function parameters regardless of the type of self. This
      // function is initially not called anywhere.
      // - Setting up $__lldb_sink, whose signature should match that of
      // $__lldb_trampoline, in number and position of parameters,
      // but which takes in type erased pointers instead (currently only the
      // pointer to self, and the pointer to the metadata). This function is
      // what is initially called by $__lldb_expr.
      // - After generating LLVM IR SwiftExpressionParser uses the sink call to
      // fish out the parameters, and redirect the call to
      // $__lldb_trampoline instead.
      // - SwiftASTManipulator also needs to make sure $__lldb_trampoline and
      // $__lldb_user_expr are generic, it does that by referring to the
      // generic parameters in the tuple argument of $__lldb_arg. This is safe
      // to do because the only purpose of $__lldb_arg is to be used by the
      // materializer to materialize the user's current environment in the lldb
      // expression.
      // FIXME: the current approach always passes in the first metadata
      // pointer, change it to allow as many metadata pointers as necessary be
      // passed in.
      // FIXME: the current approach only evaluates self as generic, make it so
      // every variable in scope can be passed in as generic, adding the
      // variables and generic parameters to the signature as demanded.
      // FIXME: the current approach names the generic parameter "T", use the
      // user's name for the generic parameter, so they can refer to it in
      // their expression.
      auto c = MakeGenericSignaturesAndCalls(local_variables, generic_sig,
                                             needs_object_ptr);
      if (!c) {
        status = Status::FromError(c.takeError());
        return status;
      }
      wrapped_stream.Printf(
          R"(
extension %s$__lldb_context {
  @LLDBDebuggerFunction %s
  %s %s {
    %s
  }
}

@LLDBDebuggerFunction %s
%s {
  do {
    $__lldb_injected_self.$__lldb_user_expr(
      $__lldb_arg
    )
  }
}


@LLDBDebuggerFunction %s
%s {
}


@LLDBDebuggerFunction %s
func $__lldb_expr(_ $__lldb_arg : UnsafeMutablePointer<Any>) {
%s
}
)",
          optional_extension, availability.c_str(), func_decorator,
          c->lldb_user_expr.c_str(), wrapped_expr_text.GetData(),
          availability.c_str(), c->lldb_trampoline.c_str(),
          availability.c_str(), c->lldb_sink.c_str(), availability.c_str(),
          c->lldb_call.c_str());

    } else {
      wrapped_stream.Printf(R"(
extension %s$__lldb_context {
  @LLDBDebuggerFunction %s
  %s func $__lldb_user_expr_%u(_ $__lldb_arg : UnsafeMutablePointer<Any>) {
    %s
  }
}
@LLDBDebuggerFunction %s
func $__lldb_expr(_ $__lldb_arg : UnsafeMutablePointer<Any>) {
  do {
    $__lldb_injected_self.$__lldb_user_expr_%u(
      $__lldb_arg
    )
  }
}
)",
                            optional_extension, availability.c_str(),
                            func_decorator, current_counter,
                            wrapped_expr_text.GetData(), availability.c_str(),
                            current_counter);
    }
  } else if (!SwiftASTManipulator::ShouldBindGenericTypes(
                 options.GetBindGenericTypes())) {
    auto c = MakeGenericSignaturesAndCalls(local_variables, generic_sig,
                                           needs_object_ptr);
    if (!c) {
      status = Status::FromError(c.takeError());
      return status;
    }
    wrapped_stream.Printf(R"(
@LLDBDebuggerFunction %s  %s {
    %s
}

@LLDBDebuggerFunction %s
%s {}


@LLDBDebuggerFunction %s
func $__lldb_expr(_ $__lldb_arg : UnsafeMutablePointer<Any>) {
  %s
}
)",
                          availability.c_str(), c->lldb_user_expr.c_str(),
                          wrapped_expr_text.GetData(), availability.c_str(),
                          c->lldb_sink.c_str(), availability.c_str(),
                          c->lldb_call.c_str());

  } else {
    wrapped_stream.Printf(
        "@LLDBDebuggerFunction %s\n"
        "func $__lldb_expr(_ $__lldb_arg : UnsafeMutablePointer<Any>) {\n"
        "%s" // This is the expression text (with newlines).
        "}\n",
        availability.c_str(), wrapped_expr_text.GetData());
  }
  return status;
}

/// Format the OS name the way that Swift availability attributes do.
static llvm::StringRef getAvailabilityName(const llvm::Triple &triple) {
    swift::LangOptions lang_options;
  lang_options.setTarget(triple);
  return swift::platformString(swift::targetPlatform(lang_options));
}

uint32_t SwiftExpressionSourceCode::GetNumBodyLines() {
  if (m_num_body_lines == 0)
    // 2 = <one for zero indexing> + <one for the body start marker>
    m_num_body_lines = 2 + std::count(m_body.begin(), m_body.end(), '\n');
  return m_num_body_lines;
}

Status SwiftExpressionSourceCode::GetText(
    std::string &text, SourceLanguage wrapping_language, bool needs_object_ptr,
    bool static_method, bool is_class, bool weak_self,
    const EvaluateExpressionOptions &options,
    const std::optional<SwiftLanguageRuntime::GenericSignature> &generic_sig,
    ExecutionContext &exe_ctx, uint32_t &first_body_line,
    llvm::ArrayRef<SwiftASTManipulator::VariableInfo> local_variables) const {
  Status status;
  Target *target = exe_ctx.GetTargetPtr();
  if (m_wrap) {
    const char *body = m_body.c_str();
    const char *pound_file = options.GetPoundLineFilePath();
    const uint32_t pound_line = options.GetPoundLineLine();
    StreamString pound_body;
    if (pound_file && pound_line) {
      if (wrapping_language.name == llvm::dwarf::DW_LNAME_Swift) {
        pound_body.Printf("#sourceLocation(file: \"%s\", line: %u)\n%s",
                          pound_file, pound_line, body);
      } else {
        pound_body.Printf("#line %u \"%s\"\n%s", pound_line, pound_file, body);
      }
      body = pound_body.GetString().data();
    }

    if (wrapping_language.name != llvm::dwarf::DW_LNAME_Swift) {
      status = Status::FromErrorString("language is not Swift");
      return status;
    }

    StreamString wrap_stream;


    // First construct a tagged form of the user expression so we can find it
    // later:
    std::string tagged_body;
    llvm::SmallString<16> buffer;
    llvm::raw_svector_ostream os_vers(buffer);

    auto arch_spec = target->GetArchitecture();
    auto triple = arch_spec.GetTriple();
    if (triple.isOSDarwin()) {
      if (auto process_sp = exe_ctx.GetProcessSP()) {
        os_vers << getAvailabilityName(triple) << " ";
        auto platform = target->GetPlatform();
        bool is_simulator = platform->GetPluginName().ends_with("-simulator");
        if (is_simulator) {
          // The simulators look like the host OS to Process, but Platform
          // can the version out of an environment variable.
          os_vers << platform->GetOSVersion(process_sp.get()).getAsString();
        } else {
          llvm::VersionTuple version = process_sp->GetHostOSVersion();
          os_vers << version.getAsString();
        }
      }
    }

    SwiftPersistentExpressionState *persistent_state =
        llvm::cast<SwiftPersistentExpressionState>(
            target->GetPersistentExpressionStateForLanguage(
                lldb::eLanguageTypeSwift));
    if (!persistent_state) {
      status = Status::FromErrorString("no persistent state");
      return status;
    }
    std::vector<CompilerDecl> persistent_results;
    // Check if we have already declared the playground stub debug functions
    persistent_state->GetSwiftPersistentDecls("__builtin_log_with_id", {},
                                              persistent_results);

    size_t num_persistent_results = persistent_results.size();
    bool need_to_declare_log_functions = num_persistent_results == 0;
    EvaluateExpressionOptions localOptions(options);

    localOptions.SetPreparePlaygroundStubFunctions(
        need_to_declare_log_functions);

    std::string full_body = m_prefix + m_body;
    status = WrapExpression(wrap_stream, full_body.c_str(), needs_object_ptr,
                            static_method, is_class, weak_self, localOptions,
                            os_vers.str(), first_body_line, local_variables,
                            generic_sig);
    if (status.Fail())
      return status;

    text = wrap_stream.GetString().str();
  } else {
    text.append(m_body);
  }

  if (!first_body_line) {
    // If this is not a playground or REPL expression, compute the
    // first line, by locating the marker. While this could be
    // determined statically, it's more future-proof to calculate it
    // here.
    uint32_t start_idx, end_idx;
    if (GetOriginalBodyBounds(text, start_idx, end_idx))
      first_body_line =
          StringRef(text.data(), text.size() - start_idx).count('\n') + 1;
  }
  return status;
}

bool SwiftExpressionSourceCode::GetOriginalBodyBounds(
    std::string transformed_text, uint32_t &start_loc, uint32_t &end_loc) {
  StringRef start_marker = GetUserCodeStartMarker();
  StringRef end_marker = GetUserCodeEndMarker();
  size_t found_loc = transformed_text.find(start_marker);
  if (found_loc == StringRef::npos ||
      found_loc > std::numeric_limits<uint32_t>::max())
    return false;
  start_loc = found_loc;

  start_loc += start_marker.size();
  found_loc = transformed_text.find(end_marker);
  if (found_loc == StringRef::npos ||
      found_loc > std::numeric_limits<uint32_t>::max())
    return false;
  end_loc = found_loc;
  return true;
}
