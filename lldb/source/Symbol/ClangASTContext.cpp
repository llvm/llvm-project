//===-- ClangASTContext.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Symbol/ClangASTContext.h"

// C Includes
// C++ Includes
#include <mutex> // std::once
#include <string>
#include <vector>

// Other libraries and framework includes

// Clang headers like to use NDEBUG inside of them to enable/disable debug 
// related features using "#ifndef NDEBUG" preprocessor blocks to do one thing
// or another. This is bad because it means that if clang was built in release
// mode, it assumes that you are building in release mode which is not always
// the case. You can end up with functions that are defined as empty in header
// files when NDEBUG is not defined, and this can cause link errors with the
// clang .a files that you have since you might be missing functions in the .a
// file. So we have to define NDEBUG when including clang headers to avoid any
// mismatches. This is covered by rdar://problem/8691220

#if !defined(NDEBUG) && !defined(LLVM_NDEBUG_OFF)
#define LLDB_DEFINED_NDEBUG_FOR_CLANG
#define NDEBUG
// Need to include assert.h so it is as clang would expect it to be (disabled)
#include <assert.h>
#endif

#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTImporter.h"
#include "clang/AST/Attr.h"
#include "clang/AST/CXXInheritance.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Mangle.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/Type.h"
#include "clang/AST/VTableBuilder.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/FileSystemOptions.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/Frontend/FrontendOptions.h"
#include "clang/Frontend/LangStandard.h"

#ifdef LLDB_DEFINED_NDEBUG_FOR_CLANG
#undef NDEBUG
#undef LLDB_DEFINED_NDEBUG_FOR_CLANG
// Need to re-include assert.h so it is as _we_ would expect it to be (enabled)
#include <assert.h>
#endif

#include "llvm/Support/Signals.h"

#include "lldb/Core/ArchSpec.h"
#include "lldb/Core/Flags.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/RegularExpression.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Core/ThreadSafeDenseMap.h"
#include "lldb/Core/UniqueCStringMap.h"
#include "Plugins/ExpressionParser/Clang/ClangUserExpression.h"
#include "Plugins/ExpressionParser/Clang/ClangFunctionCaller.h"
#include "Plugins/ExpressionParser/Clang/ClangUtilityFunction.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/ClangExternalASTSourceCallbacks.h"
#include "lldb/Symbol/ClangExternalASTSourceCommon.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/SymbolFile.h"
#include "lldb/Symbol/VerifyDecl.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Language.h"
#include "lldb/Target/ObjCLanguageRuntime.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"

#include "Plugins/SymbolFile/DWARF/DWARFASTParserClang.h"

#include <stdio.h>

#include <mutex>

using namespace lldb;
using namespace lldb_private;
using namespace llvm;
using namespace clang;

namespace
{
    static inline bool ClangASTContextSupportsLanguage (lldb::LanguageType language)
    {
        return language == eLanguageTypeUnknown || // Clang is the default type system
               Language::LanguageIsC (language) ||
               Language::LanguageIsCPlusPlus (language) ||
               Language::LanguageIsObjC (language);
    }
}

typedef lldb_private::ThreadSafeDenseMap<clang::ASTContext *, ClangASTContext*> ClangASTMap;

static ClangASTMap &
GetASTMap()
{
    static ClangASTMap *g_map_ptr = nullptr;
    static std::once_flag g_once_flag;
    std::call_once(g_once_flag,  []() {
        g_map_ptr = new ClangASTMap(); // leaked on purpose to avoid spins
    });
    return *g_map_ptr;
}


clang::AccessSpecifier
ClangASTContext::ConvertAccessTypeToAccessSpecifier (AccessType access)
{
    switch (access)
    {
    default:               break;
    case eAccessNone:      return AS_none;
    case eAccessPublic:    return AS_public;
    case eAccessPrivate:   return AS_private;
    case eAccessProtected: return AS_protected;
    }
    return AS_none;
}

static void
ParseLangArgs (LangOptions &Opts, InputKind IK, const char* triple)
{
    // FIXME: Cleanup per-file based stuff.

    // Set some properties which depend solely on the input kind; it would be nice
    // to move these to the language standard, and have the driver resolve the
    // input kind + language standard.
    if (IK == IK_Asm) {
        Opts.AsmPreprocessor = 1;
    } else if (IK == IK_ObjC ||
               IK == IK_ObjCXX ||
               IK == IK_PreprocessedObjC ||
               IK == IK_PreprocessedObjCXX) {
        Opts.ObjC1 = Opts.ObjC2 = 1;
    }

    LangStandard::Kind LangStd = LangStandard::lang_unspecified;

    if (LangStd == LangStandard::lang_unspecified) {
        // Based on the base language, pick one.
        switch (IK) {
            case IK_None:
            case IK_AST:
            case IK_LLVM_IR:
                assert (!"Invalid input kind!");
            case IK_OpenCL:
                LangStd = LangStandard::lang_opencl;
                break;
            case IK_CUDA:
            case IK_PreprocessedCuda:
                LangStd = LangStandard::lang_cuda;
                break;
            case IK_Asm:
            case IK_C:
            case IK_PreprocessedC:
            case IK_ObjC:
            case IK_PreprocessedObjC:
                LangStd = LangStandard::lang_gnu99;
                break;
            case IK_CXX:
            case IK_PreprocessedCXX:
            case IK_ObjCXX:
            case IK_PreprocessedObjCXX:
                LangStd = LangStandard::lang_gnucxx98;
                break;
        }
    }

    const LangStandard &Std = LangStandard::getLangStandardForKind(LangStd);
    Opts.LineComment = Std.hasLineComments();
    Opts.C99 = Std.isC99();
    Opts.CPlusPlus = Std.isCPlusPlus();
    Opts.CPlusPlus11 = Std.isCPlusPlus11();
    Opts.Digraphs = Std.hasDigraphs();
    Opts.GNUMode = Std.isGNUMode();
    Opts.GNUInline = !Std.isC99();
    Opts.HexFloats = Std.hasHexFloats();
    Opts.ImplicitInt = Std.hasImplicitInt();
    
    Opts.WChar = true;

    // OpenCL has some additional defaults.
    if (LangStd == LangStandard::lang_opencl) {
        Opts.OpenCL = 1;
        Opts.AltiVec = 1;
        Opts.CXXOperatorNames = 1;
        Opts.LaxVectorConversions = 1;
    }

    // OpenCL and C++ both have bool, true, false keywords.
    Opts.Bool = Opts.OpenCL || Opts.CPlusPlus;

//    if (Opts.CPlusPlus)
//        Opts.CXXOperatorNames = !Args.hasArg(OPT_fno_operator_names);
//
//    if (Args.hasArg(OPT_fobjc_gc_only))
//        Opts.setGCMode(LangOptions::GCOnly);
//    else if (Args.hasArg(OPT_fobjc_gc))
//        Opts.setGCMode(LangOptions::HybridGC);
//
//    if (Args.hasArg(OPT_print_ivar_layout))
//        Opts.ObjCGCBitmapPrint = 1;
//
//    if (Args.hasArg(OPT_faltivec))
//        Opts.AltiVec = 1;
//
//    if (Args.hasArg(OPT_pthread))
//        Opts.POSIXThreads = 1;
//
//    llvm::StringRef Vis = getLastArgValue(Args, OPT_fvisibility,
//                                          "default");
//    if (Vis == "default")
        Opts.setValueVisibilityMode(DefaultVisibility);
//    else if (Vis == "hidden")
//        Opts.setVisibilityMode(LangOptions::Hidden);
//    else if (Vis == "protected")
//        Opts.setVisibilityMode(LangOptions::Protected);
//    else
//        Diags.Report(diag::err_drv_invalid_value)
//        << Args.getLastArg(OPT_fvisibility)->getAsString(Args) << Vis;

//    Opts.OverflowChecking = Args.hasArg(OPT_ftrapv);

    // Mimicing gcc's behavior, trigraphs are only enabled if -trigraphs
    // is specified, or -std is set to a conforming mode.
    Opts.Trigraphs = !Opts.GNUMode;
//    if (Args.hasArg(OPT_trigraphs))
//        Opts.Trigraphs = 1;
//
//    Opts.DollarIdents = Args.hasFlag(OPT_fdollars_in_identifiers,
//                                     OPT_fno_dollars_in_identifiers,
//                                     !Opts.AsmPreprocessor);
//    Opts.PascalStrings = Args.hasArg(OPT_fpascal_strings);
//    Opts.Microsoft = Args.hasArg(OPT_fms_extensions);
//    Opts.WritableStrings = Args.hasArg(OPT_fwritable_strings);
//    if (Args.hasArg(OPT_fno_lax_vector_conversions))
//        Opts.LaxVectorConversions = 0;
//    Opts.Exceptions = Args.hasArg(OPT_fexceptions);
//    Opts.RTTI = !Args.hasArg(OPT_fno_rtti);
//    Opts.Blocks = Args.hasArg(OPT_fblocks);
      Opts.CharIsSigned = ArchSpec(triple).CharIsSignedByDefault();
//    Opts.ShortWChar = Args.hasArg(OPT_fshort_wchar);
//    Opts.Freestanding = Args.hasArg(OPT_ffreestanding);
//    Opts.NoBuiltin = Args.hasArg(OPT_fno_builtin) || Opts.Freestanding;
//    Opts.AssumeSaneOperatorNew = !Args.hasArg(OPT_fno_assume_sane_operator_new);
//    Opts.HeinousExtensions = Args.hasArg(OPT_fheinous_gnu_extensions);
//    Opts.AccessControl = Args.hasArg(OPT_faccess_control);
//    Opts.ElideConstructors = !Args.hasArg(OPT_fno_elide_constructors);
//    Opts.MathErrno = !Args.hasArg(OPT_fno_math_errno);
//    Opts.InstantiationDepth = getLastArgIntValue(Args, OPT_ftemplate_depth, 99,
//                                                 Diags);
//    Opts.NeXTRuntime = !Args.hasArg(OPT_fgnu_runtime);
//    Opts.ObjCConstantStringClass = getLastArgValue(Args,
//                                                   OPT_fconstant_string_class);
//    Opts.ObjCNonFragileABI = Args.hasArg(OPT_fobjc_nonfragile_abi);
//    Opts.CatchUndefined = Args.hasArg(OPT_fcatch_undefined_behavior);
//    Opts.EmitAllDecls = Args.hasArg(OPT_femit_all_decls);
//    Opts.PICLevel = getLastArgIntValue(Args, OPT_pic_level, 0, Diags);
//    Opts.Static = Args.hasArg(OPT_static_define);
    Opts.OptimizeSize = 0;

    // FIXME: Eliminate this dependency.
//    unsigned Opt =
//    Args.hasArg(OPT_Os) ? 2 : getLastArgIntValue(Args, OPT_O, 0, Diags);
//    Opts.Optimize = Opt != 0;
    unsigned Opt = 0;

    // This is the __NO_INLINE__ define, which just depends on things like the
    // optimization level and -fno-inline, not actually whether the backend has
    // inlining enabled.
    //
    // FIXME: This is affected by other options (-fno-inline).
    Opts.NoInlineDefine = !Opt;

//    unsigned SSP = getLastArgIntValue(Args, OPT_stack_protector, 0, Diags);
//    switch (SSP) {
//        default:
//            Diags.Report(diag::err_drv_invalid_value)
//            << Args.getLastArg(OPT_stack_protector)->getAsString(Args) << SSP;
//            break;
//        case 0: Opts.setStackProtectorMode(LangOptions::SSPOff); break;
//        case 1: Opts.setStackProtectorMode(LangOptions::SSPOn);  break;
//        case 2: Opts.setStackProtectorMode(LangOptions::SSPReq); break;
//    }
}


ClangASTContext::ClangASTContext (const char *target_triple) :
    TypeSystem (TypeSystem::eKindClang),
    m_target_triple (),
    m_ast_ap (),
    m_language_options_ap (),
    m_source_manager_ap (),
    m_diagnostics_engine_ap (),
    m_target_options_rp (),
    m_target_info_ap (),
    m_identifier_table_ap (),
    m_selector_table_ap (),
    m_builtins_ap (),
    m_callback_tag_decl (nullptr),
    m_callback_objc_decl (nullptr),
    m_callback_baton (nullptr),
    m_pointer_byte_size (0),
    m_ast_owned (false)
{
    if (target_triple && target_triple[0])
        SetTargetTriple (target_triple);
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
ClangASTContext::~ClangASTContext()
{
    if (m_ast_ap.get())
    {
        GetASTMap().Erase(m_ast_ap.get());
        if (!m_ast_owned)
            m_ast_ap.release();
    }

    m_builtins_ap.reset();
    m_selector_table_ap.reset();
    m_identifier_table_ap.reset();
    m_target_info_ap.reset();
    m_target_options_rp.reset();
    m_diagnostics_engine_ap.reset();
    m_source_manager_ap.reset();
    m_language_options_ap.reset();
    m_ast_ap.reset();
}

ConstString
ClangASTContext::GetPluginNameStatic()
{
    return ConstString("clang");
}

ConstString
ClangASTContext::GetPluginName()
{
    return ClangASTContext::GetPluginNameStatic();
}

uint32_t
ClangASTContext::GetPluginVersion()
{
    return 1;
}

lldb::TypeSystemSP
ClangASTContext::CreateInstance (lldb::LanguageType language,
                                 lldb_private::Module *module,
                                 Target *target)
{
    if (ClangASTContextSupportsLanguage(language))
    {
        ArchSpec arch;
        if (module)
            arch = module->GetArchitecture();
        else if (target)
            arch = target->GetArchitecture();

        if (arch.IsValid())
        {
            ArchSpec fixed_arch = arch;
            // LLVM wants this to be set to iOS or MacOSX; if we're working on
            // a bare-boards type image, change the triple for llvm's benefit.
            if (fixed_arch.GetTriple().getVendor() == llvm::Triple::Apple &&
                fixed_arch.GetTriple().getOS() == llvm::Triple::UnknownOS)
            {
                if (fixed_arch.GetTriple().getArch() == llvm::Triple::arm ||
                    fixed_arch.GetTriple().getArch() == llvm::Triple::aarch64 ||
                    fixed_arch.GetTriple().getArch() == llvm::Triple::thumb)
                {
                    fixed_arch.GetTriple().setOS(llvm::Triple::IOS);
                }
                else
                {
                    fixed_arch.GetTriple().setOS(llvm::Triple::MacOSX);
                }
            }

            if (module)
            {
                std::shared_ptr<ClangASTContext> ast_sp(new ClangASTContext);
                if (ast_sp)
                {
                    ast_sp->SetArchitecture (fixed_arch);
                }
                return ast_sp;
            }
            else if (target && target->IsValid())
            {
                std::shared_ptr<ClangASTContextForExpressions> ast_sp(new ClangASTContextForExpressions(*target));
                if (ast_sp)
                {
                    ast_sp->SetArchitecture(fixed_arch);
                    ast_sp->m_scratch_ast_source_ap.reset (new ClangASTSource(target->shared_from_this()));
                    ast_sp->m_scratch_ast_source_ap->InstallASTContext(ast_sp->getASTContext());
                    llvm::IntrusiveRefCntPtr<clang::ExternalASTSource> proxy_ast_source(ast_sp->m_scratch_ast_source_ap->CreateProxy());
                    ast_sp->SetExternalSource(proxy_ast_source);
                    return ast_sp;
                }
            }
        }
    }
    return lldb::TypeSystemSP();
}

void
ClangASTContext::EnumerateSupportedLanguages(std::set<lldb::LanguageType> &languages_for_types, std::set<lldb::LanguageType> &languages_for_expressions)
{
    static std::vector<lldb::LanguageType> s_supported_languages_for_types({
        lldb::eLanguageTypeC89,
        lldb::eLanguageTypeC,
        lldb::eLanguageTypeC11,
        lldb::eLanguageTypeC_plus_plus,
        lldb::eLanguageTypeC99,
        lldb::eLanguageTypeObjC,
        lldb::eLanguageTypeObjC_plus_plus,
        lldb::eLanguageTypeC_plus_plus_03,
        lldb::eLanguageTypeC_plus_plus_11,
        lldb::eLanguageTypeC11,
        lldb::eLanguageTypeC_plus_plus_14});
    
    static std::vector<lldb::LanguageType> s_supported_languages_for_expressions({
        lldb::eLanguageTypeC_plus_plus,
        lldb::eLanguageTypeObjC_plus_plus,
        lldb::eLanguageTypeC_plus_plus_03,
        lldb::eLanguageTypeC_plus_plus_11,
        lldb::eLanguageTypeC_plus_plus_14});

    languages_for_types.insert(s_supported_languages_for_types.begin(), s_supported_languages_for_types.end());
    languages_for_expressions.insert(s_supported_languages_for_expressions.begin(), s_supported_languages_for_expressions.end());
}


void
ClangASTContext::Initialize()
{
    PluginManager::RegisterPlugin (GetPluginNameStatic(),
                                   "clang base AST context plug-in",
                                   CreateInstance,
                                   EnumerateSupportedLanguages);
}

void
ClangASTContext::Terminate()
{
    PluginManager::UnregisterPlugin (CreateInstance);
}


void
ClangASTContext::Clear()
{
    m_ast_ap.reset();
    m_language_options_ap.reset();
    m_source_manager_ap.reset();
    m_diagnostics_engine_ap.reset();
    m_target_options_rp.reset();
    m_target_info_ap.reset();
    m_identifier_table_ap.reset();
    m_selector_table_ap.reset();
    m_builtins_ap.reset();
    m_pointer_byte_size = 0;
}

const char *
ClangASTContext::GetTargetTriple ()
{
    return m_target_triple.c_str();
}

void
ClangASTContext::SetTargetTriple (const char *target_triple)
{
    Clear();
    m_target_triple.assign(target_triple);
}

void
ClangASTContext::SetArchitecture (const ArchSpec &arch)
{
    SetTargetTriple(arch.GetTriple().str().c_str());
}

bool
ClangASTContext::HasExternalSource ()
{
    ASTContext *ast = getASTContext();
    if (ast)
        return ast->getExternalSource () != nullptr;
    return false;
}

void
ClangASTContext::SetExternalSource (llvm::IntrusiveRefCntPtr<ExternalASTSource> &ast_source_ap)
{
    ASTContext *ast = getASTContext();
    if (ast)
    {
        ast->setExternalSource (ast_source_ap);
        ast->getTranslationUnitDecl()->setHasExternalLexicalStorage(true);
        //ast->getTranslationUnitDecl()->setHasExternalVisibleStorage(true);
    }
}

void
ClangASTContext::RemoveExternalSource ()
{
    ASTContext *ast = getASTContext();
    
    if (ast)
    {
        llvm::IntrusiveRefCntPtr<ExternalASTSource> empty_ast_source_ap;
        ast->setExternalSource (empty_ast_source_ap);
        ast->getTranslationUnitDecl()->setHasExternalLexicalStorage(false);
        //ast->getTranslationUnitDecl()->setHasExternalVisibleStorage(false);
    }
}

void
ClangASTContext::setASTContext(clang::ASTContext *ast_ctx)
{
    if (!m_ast_owned) {
        m_ast_ap.release();
    }
    m_ast_owned = false;
    m_ast_ap.reset(ast_ctx);
    GetASTMap().Insert(ast_ctx, this);
}

ASTContext *
ClangASTContext::getASTContext()
{
    if (m_ast_ap.get() == nullptr)
    {
        m_ast_owned = true;
        m_ast_ap.reset(new ASTContext (*getLanguageOptions(),
                                       *getSourceManager(),
                                       *getIdentifierTable(),
                                       *getSelectorTable(),
                                       *getBuiltinContext()));
        
        m_ast_ap->getDiagnostics().setClient(getDiagnosticConsumer(), false);

        // This can be NULL if we don't know anything about the architecture or if the
        // target for an architecture isn't enabled in the llvm/clang that we built
        TargetInfo *target_info = getTargetInfo();
        if (target_info)
            m_ast_ap->InitBuiltinTypes(*target_info);
        
        if ((m_callback_tag_decl || m_callback_objc_decl) && m_callback_baton)
        {
            m_ast_ap->getTranslationUnitDecl()->setHasExternalLexicalStorage();
            //m_ast_ap->getTranslationUnitDecl()->setHasExternalVisibleStorage();
        }
        
        GetASTMap().Insert(m_ast_ap.get(), this);

        llvm::IntrusiveRefCntPtr<clang::ExternalASTSource> ast_source_ap (new ClangExternalASTSourceCallbacks (ClangASTContext::CompleteTagDecl,
                                                                                                               ClangASTContext::CompleteObjCInterfaceDecl,
                                                                                                               nullptr,
                                                                                                               ClangASTContext::LayoutRecordType,
                                                                                                               this));
        SetExternalSource (ast_source_ap);
    }
    return m_ast_ap.get();
}

ClangASTContext*
ClangASTContext::GetASTContext (clang::ASTContext* ast)
{
    ClangASTContext *clang_ast = GetASTMap().Lookup(ast);
    return clang_ast;
}

Builtin::Context *
ClangASTContext::getBuiltinContext()
{
    if (m_builtins_ap.get() == nullptr)
        m_builtins_ap.reset (new Builtin::Context());
    return m_builtins_ap.get();
}

IdentifierTable *
ClangASTContext::getIdentifierTable()
{
    if (m_identifier_table_ap.get() == nullptr)
        m_identifier_table_ap.reset(new IdentifierTable (*ClangASTContext::getLanguageOptions(), nullptr));
    return m_identifier_table_ap.get();
}

LangOptions *
ClangASTContext::getLanguageOptions()
{
    if (m_language_options_ap.get() == nullptr)
    {
        m_language_options_ap.reset(new LangOptions());
        ParseLangArgs(*m_language_options_ap, IK_ObjCXX, GetTargetTriple());
//        InitializeLangOptions(*m_language_options_ap, IK_ObjCXX);
    }
    return m_language_options_ap.get();
}

SelectorTable *
ClangASTContext::getSelectorTable()
{
    if (m_selector_table_ap.get() == nullptr)
        m_selector_table_ap.reset (new SelectorTable());
    return m_selector_table_ap.get();
}

clang::FileManager *
ClangASTContext::getFileManager()
{
    if (m_file_manager_ap.get() == nullptr)
    {
        clang::FileSystemOptions file_system_options;
        m_file_manager_ap.reset(new clang::FileManager(file_system_options));
    }
    return m_file_manager_ap.get();
}

clang::SourceManager *
ClangASTContext::getSourceManager()
{
    if (m_source_manager_ap.get() == nullptr)
        m_source_manager_ap.reset(new clang::SourceManager(*getDiagnosticsEngine(), *getFileManager()));
    return m_source_manager_ap.get();
}

clang::DiagnosticsEngine *
ClangASTContext::getDiagnosticsEngine()
{
    if (m_diagnostics_engine_ap.get() == nullptr)
    {
        llvm::IntrusiveRefCntPtr<DiagnosticIDs> diag_id_sp(new DiagnosticIDs());
        m_diagnostics_engine_ap.reset(new DiagnosticsEngine(diag_id_sp, new DiagnosticOptions()));
    }
    return m_diagnostics_engine_ap.get();
}

clang::MangleContext *
ClangASTContext::getMangleContext()
{
    if (m_mangle_ctx_ap.get() == nullptr)
        m_mangle_ctx_ap.reset (getASTContext()->createMangleContext());
    return m_mangle_ctx_ap.get();
}

class NullDiagnosticConsumer : public DiagnosticConsumer
{
public:
    NullDiagnosticConsumer ()
    {
        m_log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS);
    }
    
    void HandleDiagnostic (DiagnosticsEngine::Level DiagLevel, const Diagnostic &info)
    {
        if (m_log)
        {
            llvm::SmallVector<char, 32> diag_str(10);
            info.FormatDiagnostic(diag_str);
            diag_str.push_back('\0');
            m_log->Printf("Compiler diagnostic: %s\n", diag_str.data());
        }
    }
    
    DiagnosticConsumer *clone (DiagnosticsEngine &Diags) const
    {
        return new NullDiagnosticConsumer ();
    }
private:
    Log * m_log;
};

DiagnosticConsumer *
ClangASTContext::getDiagnosticConsumer()
{
    if (m_diagnostic_consumer_ap.get() == nullptr)
        m_diagnostic_consumer_ap.reset(new NullDiagnosticConsumer);
    
    return m_diagnostic_consumer_ap.get();
}

std::shared_ptr<TargetOptions> &
ClangASTContext::getTargetOptions() {
    if (m_target_options_rp.get() == nullptr && !m_target_triple.empty())
    {
        m_target_options_rp = std::make_shared<TargetOptions>();
        if (m_target_options_rp.get() != nullptr)
            m_target_options_rp->Triple = m_target_triple;
    }
    return m_target_options_rp;
}


TargetInfo *
ClangASTContext::getTargetInfo()
{
    // target_triple should be something like "x86_64-apple-macosx"
    if (m_target_info_ap.get() == nullptr && !m_target_triple.empty())
        m_target_info_ap.reset (TargetInfo::CreateTargetInfo(*getDiagnosticsEngine(), getTargetOptions()));
    return m_target_info_ap.get();
}

#pragma mark Basic Types

static inline bool
QualTypeMatchesBitSize(const uint64_t bit_size, ASTContext *ast, QualType qual_type)
{
    uint64_t qual_type_bit_size = ast->getTypeSize(qual_type);
    if (qual_type_bit_size == bit_size)
        return true;
    return false;
}

CompilerType
ClangASTContext::GetBuiltinTypeForEncodingAndBitSize (Encoding encoding, size_t bit_size)
{
    return ClangASTContext::GetBuiltinTypeForEncodingAndBitSize (getASTContext(), encoding, bit_size);
}

CompilerType
ClangASTContext::GetBuiltinTypeForEncodingAndBitSize (ASTContext *ast, Encoding encoding, uint32_t bit_size)
{
    if (!ast)
        return CompilerType();
    switch (encoding)
    {
    case eEncodingInvalid:
        if (QualTypeMatchesBitSize (bit_size, ast, ast->VoidPtrTy))
            return CompilerType (ast, ast->VoidPtrTy);
        break;
        
    case eEncodingUint:
        if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedCharTy))
            return CompilerType (ast, ast->UnsignedCharTy);
        if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedShortTy))
            return CompilerType (ast, ast->UnsignedShortTy);
        if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedIntTy))
            return CompilerType (ast, ast->UnsignedIntTy);
        if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedLongTy))
            return CompilerType (ast, ast->UnsignedLongTy);
        if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedLongLongTy))
            return CompilerType (ast, ast->UnsignedLongLongTy);
        if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedInt128Ty))
            return CompilerType (ast, ast->UnsignedInt128Ty);
        break;
        
    case eEncodingSint:
        if (QualTypeMatchesBitSize (bit_size, ast, ast->CharTy))
            return CompilerType (ast, ast->CharTy);
        if (QualTypeMatchesBitSize (bit_size, ast, ast->ShortTy))
            return CompilerType (ast, ast->ShortTy);
        if (QualTypeMatchesBitSize (bit_size, ast, ast->IntTy))
            return CompilerType (ast, ast->IntTy);
        if (QualTypeMatchesBitSize (bit_size, ast, ast->LongTy))
            return CompilerType (ast, ast->LongTy);
        if (QualTypeMatchesBitSize (bit_size, ast, ast->LongLongTy))
            return CompilerType (ast, ast->LongLongTy);
        if (QualTypeMatchesBitSize (bit_size, ast, ast->Int128Ty))
            return CompilerType (ast, ast->Int128Ty);
        break;
        
    case eEncodingIEEE754:
        if (QualTypeMatchesBitSize (bit_size, ast, ast->FloatTy))
            return CompilerType (ast, ast->FloatTy);
        if (QualTypeMatchesBitSize (bit_size, ast, ast->DoubleTy))
            return CompilerType (ast, ast->DoubleTy);
        if (QualTypeMatchesBitSize (bit_size, ast, ast->LongDoubleTy))
            return CompilerType (ast, ast->LongDoubleTy);
        if (QualTypeMatchesBitSize (bit_size, ast, ast->HalfTy))
            return CompilerType (ast, ast->HalfTy);
        break;
        
    case eEncodingVector:
        // Sanity check that bit_size is a multiple of 8's.
        if (bit_size && !(bit_size & 0x7u))
            return CompilerType (ast, ast->getExtVectorType (ast->UnsignedCharTy, bit_size/8));
        break;
    }
    
    return CompilerType();
}



lldb::BasicType
ClangASTContext::GetBasicTypeEnumeration (const ConstString &name)
{
    if (name)
    {
        typedef UniqueCStringMap<lldb::BasicType> TypeNameToBasicTypeMap;
        static TypeNameToBasicTypeMap g_type_map;
        static std::once_flag g_once_flag;
        std::call_once(g_once_flag, [](){
            // "void"
            g_type_map.Append(ConstString("void").GetCString(), eBasicTypeVoid);
            
            // "char"
            g_type_map.Append(ConstString("char").GetCString(), eBasicTypeChar);
            g_type_map.Append(ConstString("signed char").GetCString(), eBasicTypeSignedChar);
            g_type_map.Append(ConstString("unsigned char").GetCString(), eBasicTypeUnsignedChar);
            g_type_map.Append(ConstString("wchar_t").GetCString(), eBasicTypeWChar);
            g_type_map.Append(ConstString("signed wchar_t").GetCString(), eBasicTypeSignedWChar);
            g_type_map.Append(ConstString("unsigned wchar_t").GetCString(), eBasicTypeUnsignedWChar);
            // "short"
            g_type_map.Append(ConstString("short").GetCString(), eBasicTypeShort);
            g_type_map.Append(ConstString("short int").GetCString(), eBasicTypeShort);
            g_type_map.Append(ConstString("unsigned short").GetCString(), eBasicTypeUnsignedShort);
            g_type_map.Append(ConstString("unsigned short int").GetCString(), eBasicTypeUnsignedShort);
            
            // "int"
            g_type_map.Append(ConstString("int").GetCString(), eBasicTypeInt);
            g_type_map.Append(ConstString("signed int").GetCString(), eBasicTypeInt);
            g_type_map.Append(ConstString("unsigned int").GetCString(), eBasicTypeUnsignedInt);
            g_type_map.Append(ConstString("unsigned").GetCString(), eBasicTypeUnsignedInt);
            
            // "long"
            g_type_map.Append(ConstString("long").GetCString(), eBasicTypeLong);
            g_type_map.Append(ConstString("long int").GetCString(), eBasicTypeLong);
            g_type_map.Append(ConstString("unsigned long").GetCString(), eBasicTypeUnsignedLong);
            g_type_map.Append(ConstString("unsigned long int").GetCString(), eBasicTypeUnsignedLong);
            
            // "long long"
            g_type_map.Append(ConstString("long long").GetCString(), eBasicTypeLongLong);
            g_type_map.Append(ConstString("long long int").GetCString(), eBasicTypeLongLong);
            g_type_map.Append(ConstString("unsigned long long").GetCString(), eBasicTypeUnsignedLongLong);
            g_type_map.Append(ConstString("unsigned long long int").GetCString(), eBasicTypeUnsignedLongLong);
            
            // "int128"
            g_type_map.Append(ConstString("__int128_t").GetCString(), eBasicTypeInt128);
            g_type_map.Append(ConstString("__uint128_t").GetCString(), eBasicTypeUnsignedInt128);
            
            // Miscellaneous
            g_type_map.Append(ConstString("bool").GetCString(), eBasicTypeBool);
            g_type_map.Append(ConstString("float").GetCString(), eBasicTypeFloat);
            g_type_map.Append(ConstString("double").GetCString(), eBasicTypeDouble);
            g_type_map.Append(ConstString("long double").GetCString(), eBasicTypeLongDouble);
            g_type_map.Append(ConstString("id").GetCString(), eBasicTypeObjCID);
            g_type_map.Append(ConstString("SEL").GetCString(), eBasicTypeObjCSel);
            g_type_map.Append(ConstString("nullptr").GetCString(), eBasicTypeNullPtr);
            g_type_map.Sort();
        });
        
        return g_type_map.Find(name.GetCString(), eBasicTypeInvalid);
    }
    return eBasicTypeInvalid;
}

CompilerType
ClangASTContext::GetBasicType (ASTContext *ast, const ConstString &name)
{
    if (ast)
    {
        lldb::BasicType basic_type = ClangASTContext::GetBasicTypeEnumeration (name);
        return ClangASTContext::GetBasicType (ast, basic_type);
    }
    return CompilerType();
}

uint32_t
ClangASTContext::GetPointerByteSize ()
{
    if (m_pointer_byte_size == 0)
        m_pointer_byte_size = GetBasicType(lldb::eBasicTypeVoid).GetPointerType().GetByteSize(nullptr);
    return m_pointer_byte_size;
}

CompilerType
ClangASTContext::GetBasicType (lldb::BasicType basic_type)
{
    return GetBasicType (getASTContext(), basic_type);
}

CompilerType
ClangASTContext::GetBasicType (ASTContext *ast, lldb::BasicType basic_type)
{
    if (ast)
    {
        lldb::opaque_compiler_type_t clang_type = nullptr;
        
        switch (basic_type)
        {
            case eBasicTypeInvalid:
            case eBasicTypeOther:
                break;
            case eBasicTypeVoid:
                clang_type = ast->VoidTy.getAsOpaquePtr();
                break;
            case eBasicTypeChar:
                clang_type = ast->CharTy.getAsOpaquePtr();
                break;
            case eBasicTypeSignedChar:
                clang_type = ast->SignedCharTy.getAsOpaquePtr();
                break;
            case eBasicTypeUnsignedChar:
                clang_type = ast->UnsignedCharTy.getAsOpaquePtr();
                break;
            case eBasicTypeWChar:
                clang_type = ast->getWCharType().getAsOpaquePtr();
                break;
            case eBasicTypeSignedWChar:
                clang_type = ast->getSignedWCharType().getAsOpaquePtr();
                break;
            case eBasicTypeUnsignedWChar:
                clang_type = ast->getUnsignedWCharType().getAsOpaquePtr();
                break;
            case eBasicTypeChar16:
                clang_type = ast->Char16Ty.getAsOpaquePtr();
                break;
            case eBasicTypeChar32:
                clang_type = ast->Char32Ty.getAsOpaquePtr();
                break;
            case eBasicTypeShort:
                clang_type = ast->ShortTy.getAsOpaquePtr();
                break;
            case eBasicTypeUnsignedShort:
                clang_type = ast->UnsignedShortTy.getAsOpaquePtr();
                break;
            case eBasicTypeInt:
                clang_type = ast->IntTy.getAsOpaquePtr();
                break;
            case eBasicTypeUnsignedInt:
                clang_type = ast->UnsignedIntTy.getAsOpaquePtr();
                break;
            case eBasicTypeLong:
                clang_type = ast->LongTy.getAsOpaquePtr();
                break;
            case eBasicTypeUnsignedLong:
                clang_type = ast->UnsignedLongTy.getAsOpaquePtr();
                break;
            case eBasicTypeLongLong:
                clang_type = ast->LongLongTy.getAsOpaquePtr();
                break;
            case eBasicTypeUnsignedLongLong:
                clang_type = ast->UnsignedLongLongTy.getAsOpaquePtr();
                break;
            case eBasicTypeInt128:
                clang_type = ast->Int128Ty.getAsOpaquePtr();
                break;
            case eBasicTypeUnsignedInt128:
                clang_type = ast->UnsignedInt128Ty.getAsOpaquePtr();
                break;
            case eBasicTypeBool:
                clang_type = ast->BoolTy.getAsOpaquePtr();
                break;
            case eBasicTypeHalf:
                clang_type = ast->HalfTy.getAsOpaquePtr();
                break;
            case eBasicTypeFloat:
                clang_type = ast->FloatTy.getAsOpaquePtr();
                break;
            case eBasicTypeDouble:
                clang_type = ast->DoubleTy.getAsOpaquePtr();
                break;
            case eBasicTypeLongDouble:
                clang_type = ast->LongDoubleTy.getAsOpaquePtr();
                break;
            case eBasicTypeFloatComplex:
                clang_type = ast->FloatComplexTy.getAsOpaquePtr();
                break;
            case eBasicTypeDoubleComplex:
                clang_type = ast->DoubleComplexTy.getAsOpaquePtr();
                break;
            case eBasicTypeLongDoubleComplex:
                clang_type = ast->LongDoubleComplexTy.getAsOpaquePtr();
                break;
            case eBasicTypeObjCID:
                clang_type = ast->getObjCIdType().getAsOpaquePtr();
                break;
            case eBasicTypeObjCClass:
                clang_type = ast->getObjCClassType().getAsOpaquePtr();
                break;
            case eBasicTypeObjCSel:
                clang_type = ast->getObjCSelType().getAsOpaquePtr();
                break;
            case eBasicTypeNullPtr:
                clang_type = ast->NullPtrTy.getAsOpaquePtr();
                break;
        }
        
        if (clang_type)
            return CompilerType (GetASTContext(ast), clang_type);
    }
    return CompilerType();
}


CompilerType
ClangASTContext::GetBuiltinTypeForDWARFEncodingAndBitSize (const char *type_name, uint32_t dw_ate, uint32_t bit_size)
{
    ASTContext *ast = getASTContext();
    
#define streq(a,b) strcmp(a,b) == 0
    assert (ast != nullptr);
    if (ast)
    {
        switch (dw_ate)
        {
            default:
                break;
                
            case DW_ATE_address:
                if (QualTypeMatchesBitSize (bit_size, ast, ast->VoidPtrTy))
                    return CompilerType (ast, ast->VoidPtrTy);
                break;
                
            case DW_ATE_boolean:
                if (QualTypeMatchesBitSize (bit_size, ast, ast->BoolTy))
                    return CompilerType (ast, ast->BoolTy);
                if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedCharTy))
                    return CompilerType (ast, ast->UnsignedCharTy);
                if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedShortTy))
                    return CompilerType (ast, ast->UnsignedShortTy);
                if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedIntTy))
                    return CompilerType (ast, ast->UnsignedIntTy);
                break;
                
            case DW_ATE_lo_user:
                // This has been seen to mean DW_AT_complex_integer
                if (type_name)
                {
                    if (::strstr(type_name, "complex"))
                    {
                        CompilerType complex_int_clang_type = GetBuiltinTypeForDWARFEncodingAndBitSize ("int", DW_ATE_signed, bit_size/2);
                        return CompilerType (ast, ast->getComplexType (GetQualType(complex_int_clang_type)));
                    }
                }
                break;
                
            case DW_ATE_complex_float:
                if (QualTypeMatchesBitSize (bit_size, ast, ast->FloatComplexTy))
                    return CompilerType (ast, ast->FloatComplexTy);
                else if (QualTypeMatchesBitSize (bit_size, ast, ast->DoubleComplexTy))
                    return CompilerType (ast, ast->DoubleComplexTy);
                else if (QualTypeMatchesBitSize (bit_size, ast, ast->LongDoubleComplexTy))
                    return CompilerType (ast, ast->LongDoubleComplexTy);
                else 
                {
                    CompilerType complex_float_clang_type = GetBuiltinTypeForDWARFEncodingAndBitSize ("float", DW_ATE_float, bit_size/2);
                    return CompilerType (ast, ast->getComplexType (GetQualType(complex_float_clang_type)));
                }
                break;
                
            case DW_ATE_float:
                if (streq(type_name, "float") && QualTypeMatchesBitSize (bit_size, ast, ast->FloatTy))
                    return CompilerType (ast, ast->FloatTy);
                if (streq(type_name, "double") && QualTypeMatchesBitSize (bit_size, ast, ast->DoubleTy))
                    return CompilerType (ast, ast->DoubleTy);
                if (streq(type_name, "long double") && QualTypeMatchesBitSize (bit_size, ast, ast->LongDoubleTy))
                    return CompilerType (ast, ast->LongDoubleTy);
                // Fall back to not requiring a name match
                if (QualTypeMatchesBitSize (bit_size, ast, ast->FloatTy))
                    return CompilerType (ast, ast->FloatTy);
                if (QualTypeMatchesBitSize (bit_size, ast, ast->DoubleTy))
                    return CompilerType (ast, ast->DoubleTy);
                if (QualTypeMatchesBitSize (bit_size, ast, ast->LongDoubleTy))
                    return CompilerType (ast, ast->LongDoubleTy);
                if (QualTypeMatchesBitSize (bit_size, ast, ast->HalfTy))
                    return CompilerType (ast, ast->HalfTy);
                break;
                
            case DW_ATE_signed:
                if (type_name)
                {
                    if (streq(type_name, "wchar_t") &&
                        QualTypeMatchesBitSize (bit_size, ast, ast->WCharTy) &&
                        (getTargetInfo() && TargetInfo::isTypeSigned (getTargetInfo()->getWCharType())))
                        return CompilerType (ast, ast->WCharTy);
                    if (streq(type_name, "void") &&
                        QualTypeMatchesBitSize (bit_size, ast, ast->VoidTy))
                        return CompilerType (ast, ast->VoidTy);
                    if (strstr(type_name, "long long") &&
                        QualTypeMatchesBitSize (bit_size, ast, ast->LongLongTy))
                        return CompilerType (ast, ast->LongLongTy);
                    if (strstr(type_name, "long") &&
                        QualTypeMatchesBitSize (bit_size, ast, ast->LongTy))
                        return CompilerType (ast, ast->LongTy);
                    if (strstr(type_name, "short") &&
                        QualTypeMatchesBitSize (bit_size, ast, ast->ShortTy))
                        return CompilerType (ast, ast->ShortTy);
                    if (strstr(type_name, "char"))
                    {
                        if (QualTypeMatchesBitSize (bit_size, ast, ast->CharTy))
                            return CompilerType (ast, ast->CharTy);
                        if (QualTypeMatchesBitSize (bit_size, ast, ast->SignedCharTy))
                            return CompilerType (ast, ast->SignedCharTy);
                    }
                    if (strstr(type_name, "int"))
                    {
                        if (QualTypeMatchesBitSize (bit_size, ast, ast->IntTy))
                            return CompilerType (ast, ast->IntTy);
                        if (QualTypeMatchesBitSize (bit_size, ast, ast->Int128Ty))
                            return CompilerType (ast, ast->Int128Ty);
                    }
                }
                // We weren't able to match up a type name, just search by size
                if (QualTypeMatchesBitSize (bit_size, ast, ast->CharTy))
                    return CompilerType (ast, ast->CharTy);
                if (QualTypeMatchesBitSize (bit_size, ast, ast->ShortTy))
                    return CompilerType (ast, ast->ShortTy);
                if (QualTypeMatchesBitSize (bit_size, ast, ast->IntTy))
                    return CompilerType (ast, ast->IntTy);
                if (QualTypeMatchesBitSize (bit_size, ast, ast->LongTy))
                    return CompilerType (ast, ast->LongTy);
                if (QualTypeMatchesBitSize (bit_size, ast, ast->LongLongTy))
                    return CompilerType (ast, ast->LongLongTy);
                if (QualTypeMatchesBitSize (bit_size, ast, ast->Int128Ty))
                    return CompilerType (ast, ast->Int128Ty);
                break;

            case DW_ATE_signed_char:
                if (ast->getLangOpts().CharIsSigned && type_name && streq(type_name, "char"))
                {
                    if (QualTypeMatchesBitSize (bit_size, ast, ast->CharTy))
                        return CompilerType (ast, ast->CharTy);
                }
                if (QualTypeMatchesBitSize (bit_size, ast, ast->SignedCharTy))
                    return CompilerType (ast, ast->SignedCharTy);
                break;
                
            case DW_ATE_unsigned:
                if (type_name)
                {
                    if (streq(type_name, "wchar_t"))
                    {
                        if (QualTypeMatchesBitSize (bit_size, ast, ast->WCharTy))
                        {
                            if (!(getTargetInfo() && TargetInfo::isTypeSigned (getTargetInfo()->getWCharType())))
                                return CompilerType (ast, ast->WCharTy);
                        }
                    }
                    if (strstr(type_name, "long long"))
                    {
                        if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedLongLongTy))
                            return CompilerType (ast, ast->UnsignedLongLongTy);
                    }
                    else if (strstr(type_name, "long"))
                    {
                        if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedLongTy))
                            return CompilerType (ast, ast->UnsignedLongTy);
                    }
                    else if (strstr(type_name, "short"))
                    {
                        if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedShortTy))
                            return CompilerType (ast, ast->UnsignedShortTy);
                    }
                    else if (strstr(type_name, "char"))
                    {
                        if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedCharTy))
                            return CompilerType (ast, ast->UnsignedCharTy);
                    }
                    else if (strstr(type_name, "int"))
                    {
                        if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedIntTy))
                            return CompilerType (ast, ast->UnsignedIntTy);
                        if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedInt128Ty))
                            return CompilerType (ast, ast->UnsignedInt128Ty);
                    }
                }
                // We weren't able to match up a type name, just search by size
                if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedCharTy))
                    return CompilerType (ast, ast->UnsignedCharTy);
                if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedShortTy))
                    return CompilerType (ast, ast->UnsignedShortTy);
                if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedIntTy))
                    return CompilerType (ast, ast->UnsignedIntTy);
                if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedLongTy))
                    return CompilerType (ast, ast->UnsignedLongTy);
                if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedLongLongTy))
                    return CompilerType (ast, ast->UnsignedLongLongTy);
                if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedInt128Ty))
                    return CompilerType (ast, ast->UnsignedInt128Ty);
                break;

            case DW_ATE_unsigned_char:
                if (!ast->getLangOpts().CharIsSigned && type_name && streq(type_name, "char"))
                {
                    if (QualTypeMatchesBitSize (bit_size, ast, ast->CharTy))
                        return CompilerType (ast, ast->CharTy);
                }
                if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedCharTy))
                    return CompilerType (ast, ast->UnsignedCharTy);
                if (QualTypeMatchesBitSize (bit_size, ast, ast->UnsignedShortTy))
                    return CompilerType (ast, ast->UnsignedShortTy);
                break;
                
            case DW_ATE_imaginary_float:
                break;
                
            case DW_ATE_UTF:
                if (type_name)
                {
                    if (streq(type_name, "char16_t"))
                    {
                        return CompilerType (ast, ast->Char16Ty);
                    }
                    else if (streq(type_name, "char32_t"))
                    {
                        return CompilerType (ast, ast->Char32Ty);
                    }
                }
                break;
        }
    }
    // This assert should fire for anything that we don't catch above so we know
    // to fix any issues we run into.
    if (type_name)
    {
        Host::SystemLog (Host::eSystemLogError, "error: need to add support for DW_TAG_base_type '%s' encoded with DW_ATE = 0x%x, bit_size = %u\n", type_name, dw_ate, bit_size);
    }
    else
    {
        Host::SystemLog (Host::eSystemLogError, "error: need to add support for DW_TAG_base_type encoded with DW_ATE = 0x%x, bit_size = %u\n", dw_ate, bit_size);
    }
    return CompilerType ();
}

CompilerType
ClangASTContext::GetUnknownAnyType(clang::ASTContext *ast)
{
    if (ast)
        return CompilerType (ast, ast->UnknownAnyTy);
    return CompilerType();
}

CompilerType
ClangASTContext::GetCStringType (bool is_const)
{
    ASTContext *ast = getASTContext();
    QualType char_type(ast->CharTy);
    
    if (is_const)
        char_type.addConst();
    
    return CompilerType (ast, ast->getPointerType(char_type));
}

clang::DeclContext *
ClangASTContext::GetTranslationUnitDecl (clang::ASTContext *ast)
{
    return ast->getTranslationUnitDecl();
}

clang::Decl *
ClangASTContext::CopyDecl (ASTContext *dst_ast, 
                           ASTContext *src_ast,
                           clang::Decl *source_decl)
{    
    FileSystemOptions file_system_options;
    FileManager file_manager (file_system_options);
    ASTImporter importer(*dst_ast, file_manager,
                         *src_ast, file_manager,
                         false);
    
    return importer.Import(source_decl);
}

bool
ClangASTContext::AreTypesSame (CompilerType type1,
                               CompilerType type2,
                               bool ignore_qualifiers)
{
    ClangASTContext *ast = llvm::dyn_cast_or_null<ClangASTContext>(type1.GetTypeSystem());
    if (!ast || ast != type2.GetTypeSystem())
        return false;

    if (type1.GetOpaqueQualType() == type2.GetOpaqueQualType())
        return true;

    QualType type1_qual = GetQualType(type1);
    QualType type2_qual = GetQualType(type2);
    
    if (ignore_qualifiers)
    {
        type1_qual = type1_qual.getUnqualifiedType();
        type2_qual = type2_qual.getUnqualifiedType();
    }
    
    return ast->getASTContext()->hasSameType (type1_qual, type2_qual);
}

CompilerType
ClangASTContext::GetTypeForDecl (clang::NamedDecl *decl)
{
    if (clang::ObjCInterfaceDecl *interface_decl = llvm::dyn_cast<clang::ObjCInterfaceDecl>(decl))
        return GetTypeForDecl(interface_decl);
    if (clang::TagDecl *tag_decl = llvm::dyn_cast<clang::TagDecl>(decl))
        return GetTypeForDecl(tag_decl);
    return CompilerType();
}


CompilerType
ClangASTContext::GetTypeForDecl (TagDecl *decl)
{
    // No need to call the getASTContext() accessor (which can create the AST
    // if it isn't created yet, because we can't have created a decl in this
    // AST if our AST didn't already exist...
    ASTContext *ast = &decl->getASTContext();
    if (ast)
        return CompilerType (ast, ast->getTagDeclType(decl));
    return CompilerType();
}

CompilerType
ClangASTContext::GetTypeForDecl (ObjCInterfaceDecl *decl)
{
    // No need to call the getASTContext() accessor (which can create the AST
    // if it isn't created yet, because we can't have created a decl in this
    // AST if our AST didn't already exist...
    ASTContext *ast = &decl->getASTContext();
    if (ast)
        return CompilerType (ast, ast->getObjCInterfaceType(decl));
    return CompilerType();
}

#pragma mark Structure, Unions, Classes

CompilerType
ClangASTContext::CreateRecordType (DeclContext *decl_ctx,
                                   AccessType access_type,
                                   const char *name,
                                   int kind,
                                   LanguageType language,
                                   ClangASTMetadata *metadata)
{
    ASTContext *ast = getASTContext();
    assert (ast != nullptr);
     
    if (decl_ctx == nullptr)
        decl_ctx = ast->getTranslationUnitDecl();


    if (language == eLanguageTypeObjC || language == eLanguageTypeObjC_plus_plus)
    {
        bool isForwardDecl = true;
        bool isInternal = false;
        return CreateObjCClass (name, decl_ctx, isForwardDecl, isInternal, metadata);
    }

    // NOTE: Eventually CXXRecordDecl will be merged back into RecordDecl and
    // we will need to update this code. I was told to currently always use
    // the CXXRecordDecl class since we often don't know from debug information
    // if something is struct or a class, so we default to always use the more
    // complete definition just in case.
    
    bool is_anonymous = (!name) || (!name[0]);
    
    CXXRecordDecl *decl = CXXRecordDecl::Create (*ast,
                                                 (TagDecl::TagKind)kind,
                                                 decl_ctx,
                                                 SourceLocation(),
                                                 SourceLocation(),
                                                 is_anonymous ? nullptr : &ast->Idents.get(name));
    
    if (is_anonymous)
        decl->setAnonymousStructOrUnion(true);
    
    if (decl)
    {
        if (metadata)
            SetMetadata(ast, decl, *metadata);

        if (access_type != eAccessNone)
            decl->setAccess (ConvertAccessTypeToAccessSpecifier (access_type));
    
        if (decl_ctx)
            decl_ctx->addDecl (decl);

        return CompilerType(ast, ast->getTagDeclType(decl));
    }
    return CompilerType();
}

static TemplateParameterList *
CreateTemplateParameterList (ASTContext *ast, 
                             const ClangASTContext::TemplateParameterInfos &template_param_infos,
                             llvm::SmallVector<NamedDecl *, 8> &template_param_decls)
{
    const bool parameter_pack = false;
    const bool is_typename = false;
    const unsigned depth = 0;
    const size_t num_template_params = template_param_infos.GetSize();
    for (size_t i=0; i<num_template_params; ++i)
    {
        const char *name = template_param_infos.names[i];
        
        IdentifierInfo *identifier_info = nullptr;
        if (name && name[0])
            identifier_info = &ast->Idents.get(name);
        if (template_param_infos.args[i].getKind() == TemplateArgument::Integral)
        {
            template_param_decls.push_back (NonTypeTemplateParmDecl::Create (*ast,
                                                                             ast->getTranslationUnitDecl(), // Is this the right decl context?, SourceLocation StartLoc,
                                                                             SourceLocation(), 
                                                                             SourceLocation(), 
                                                                             depth, 
                                                                             i,
                                                                             identifier_info,
                                                                             template_param_infos.args[i].getIntegralType(), 
                                                                             parameter_pack, 
                                                                             nullptr));
            
        }
        else
        {
            template_param_decls.push_back (TemplateTypeParmDecl::Create (*ast, 
                                                                          ast->getTranslationUnitDecl(), // Is this the right decl context?
                                                                          SourceLocation(),
                                                                          SourceLocation(),
                                                                          depth, 
                                                                          i,
                                                                          identifier_info,
                                                                          is_typename,
                                                                          parameter_pack));
        }
    }

    TemplateParameterList *template_param_list = TemplateParameterList::Create (*ast,
                                                                                SourceLocation(),
                                                                                SourceLocation(),
                                                                                &template_param_decls.front(),
                                                                                template_param_decls.size(),
                                                                                SourceLocation());
    return template_param_list;
}

clang::FunctionTemplateDecl *
ClangASTContext::CreateFunctionTemplateDecl (clang::DeclContext *decl_ctx,
                                             clang::FunctionDecl *func_decl,
                                             const char *name, 
                                             const TemplateParameterInfos &template_param_infos)
{
//    /// \brief Create a function template node.
    ASTContext *ast = getASTContext();
    
    llvm::SmallVector<NamedDecl *, 8> template_param_decls;

    TemplateParameterList *template_param_list = CreateTemplateParameterList (ast,
                                                                              template_param_infos, 
                                                                              template_param_decls);
    FunctionTemplateDecl *func_tmpl_decl = FunctionTemplateDecl::Create (*ast,
                                                                         decl_ctx,
                                                                         func_decl->getLocation(),
                                                                         func_decl->getDeclName(),
                                                                         template_param_list,
                                                                         func_decl);
    
    for (size_t i=0, template_param_decl_count = template_param_decls.size();
         i < template_param_decl_count;
         ++i)
    {
        // TODO: verify which decl context we should put template_param_decls into..
        template_param_decls[i]->setDeclContext (func_decl); 
    }

    return func_tmpl_decl;
}

void
ClangASTContext::CreateFunctionTemplateSpecializationInfo (FunctionDecl *func_decl, 
                                                           clang::FunctionTemplateDecl *func_tmpl_decl,
                                                           const TemplateParameterInfos &infos)
{
    TemplateArgumentList template_args (TemplateArgumentList::OnStack,
                                        infos.args.data(), 
                                        infos.args.size());

    func_decl->setFunctionTemplateSpecialization (func_tmpl_decl,
                                                  &template_args,
                                                  nullptr);
}


ClassTemplateDecl *
ClangASTContext::CreateClassTemplateDecl (DeclContext *decl_ctx,
                                          lldb::AccessType access_type,
                                          const char *class_name, 
                                          int kind, 
                                          const TemplateParameterInfos &template_param_infos)
{
    ASTContext *ast = getASTContext();
    
    ClassTemplateDecl *class_template_decl = nullptr;
    if (decl_ctx == nullptr)
        decl_ctx = ast->getTranslationUnitDecl();
    
    IdentifierInfo &identifier_info = ast->Idents.get(class_name);
    DeclarationName decl_name (&identifier_info);

    clang::DeclContext::lookup_result result = decl_ctx->lookup(decl_name);
    
    for (NamedDecl *decl : result)
    {
        class_template_decl = dyn_cast<clang::ClassTemplateDecl>(decl);
        if (class_template_decl)
            return class_template_decl;
    }

    llvm::SmallVector<NamedDecl *, 8> template_param_decls;

    TemplateParameterList *template_param_list = CreateTemplateParameterList (ast,
                                                                              template_param_infos, 
                                                                              template_param_decls);

    CXXRecordDecl *template_cxx_decl = CXXRecordDecl::Create (*ast,
                                                              (TagDecl::TagKind)kind,
                                                              decl_ctx,  // What decl context do we use here? TU? The actual decl context?
                                                              SourceLocation(),
                                                              SourceLocation(),
                                                              &identifier_info);

    for (size_t i=0, template_param_decl_count = template_param_decls.size();
         i < template_param_decl_count;
         ++i)
    {
        template_param_decls[i]->setDeclContext (template_cxx_decl);
    }

    // With templated classes, we say that a class is templated with
    // specializations, but that the bare class has no functions.
    //template_cxx_decl->startDefinition();
    //template_cxx_decl->completeDefinition();
    
    class_template_decl = ClassTemplateDecl::Create (*ast,
                                                     decl_ctx,  // What decl context do we use here? TU? The actual decl context?
                                                     SourceLocation(),
                                                     decl_name,
                                                     template_param_list,
                                                     template_cxx_decl,
                                                     nullptr);
    
    if (class_template_decl)
    {
        if (access_type != eAccessNone)
            class_template_decl->setAccess (ConvertAccessTypeToAccessSpecifier (access_type));
        
        //if (TagDecl *ctx_tag_decl = dyn_cast<TagDecl>(decl_ctx))
        //    CompleteTagDeclarationDefinition(GetTypeForDecl(ctx_tag_decl));
        
        decl_ctx->addDecl (class_template_decl);
        
#ifdef LLDB_CONFIGURATION_DEBUG
        VerifyDecl(class_template_decl);
#endif
    }

    return class_template_decl;
}


ClassTemplateSpecializationDecl *
ClangASTContext::CreateClassTemplateSpecializationDecl (DeclContext *decl_ctx,
                                                        ClassTemplateDecl *class_template_decl,
                                                        int kind,
                                                        const TemplateParameterInfos &template_param_infos)
{
    ASTContext *ast = getASTContext();
    ClassTemplateSpecializationDecl *class_template_specialization_decl = ClassTemplateSpecializationDecl::Create (*ast, 
                                                                                                                   (TagDecl::TagKind)kind,
                                                                                                                   decl_ctx,
                                                                                                                   SourceLocation(), 
                                                                                                                   SourceLocation(),
                                                                                                                   class_template_decl,
                                                                                                                   &template_param_infos.args.front(),
                                                                                                                   template_param_infos.args.size(),
                                                                                                                   nullptr);
    
    class_template_specialization_decl->setSpecializationKind(TSK_ExplicitSpecialization);
    
    return class_template_specialization_decl;
}

CompilerType
ClangASTContext::CreateClassTemplateSpecializationType (ClassTemplateSpecializationDecl *class_template_specialization_decl)
{
    if (class_template_specialization_decl)
    {
        ASTContext *ast = getASTContext();
        if (ast)
            return CompilerType(ast, ast->getTagDeclType(class_template_specialization_decl));
    }
    return CompilerType();
}

static inline bool
check_op_param (uint32_t op_kind, bool unary, bool binary, uint32_t num_params)
{
    // Special-case call since it can take any number of operands
    if(op_kind == OO_Call)
        return true;
    
    // The parameter count doesn't include "this"
    if (num_params == 0)
        return unary;
    if (num_params == 1)
        return binary;
    else 
    return false;
}

bool
ClangASTContext::CheckOverloadedOperatorKindParameterCount (uint32_t op_kind, uint32_t num_params)
{
    switch (op_kind)
    {
    default:
        break;
    // C++ standard allows any number of arguments to new/delete
    case OO_New:
    case OO_Array_New:
    case OO_Delete:
    case OO_Array_Delete:
        return true;
    }
    
#define OVERLOADED_OPERATOR(Name,Spelling,Token,Unary,Binary,MemberOnly) case OO_##Name: return check_op_param (op_kind, Unary, Binary, num_params);
    switch (op_kind)
    {
#include "clang/Basic/OperatorKinds.def"
        default: break;
    }
    return false;
}

clang::AccessSpecifier
ClangASTContext::UnifyAccessSpecifiers (clang::AccessSpecifier lhs, clang::AccessSpecifier rhs)
{
    clang::AccessSpecifier ret = lhs;
    
    // Make the access equal to the stricter of the field and the nested field's access
    switch (ret)
    {
        case clang::AS_none:
            break;
        case clang::AS_private:
            break;
        case clang::AS_protected:
            if (rhs == AS_private)
                ret = AS_private;
            break;
        case clang::AS_public:
            ret = rhs;
            break;
    }
    
    return ret;
}

bool
ClangASTContext::FieldIsBitfield (FieldDecl* field, uint32_t& bitfield_bit_size)
{
    return FieldIsBitfield(getASTContext(), field, bitfield_bit_size);
}

bool
ClangASTContext::FieldIsBitfield
(
    ASTContext *ast,
    FieldDecl* field,
    uint32_t& bitfield_bit_size
)
{
    if (ast == nullptr || field == nullptr)
        return false;

    if (field->isBitField())
    {
        Expr* bit_width_expr = field->getBitWidth();
        if (bit_width_expr)
        {
            llvm::APSInt bit_width_apsint;
            if (bit_width_expr->isIntegerConstantExpr(bit_width_apsint, *ast))
            {
                bitfield_bit_size = bit_width_apsint.getLimitedValue(UINT32_MAX);
                return true;
            }
        }
    }
    return false;
}

bool
ClangASTContext::RecordHasFields (const RecordDecl *record_decl)
{
    if (record_decl == nullptr)
        return false;

    if (!record_decl->field_empty())
        return true;

    // No fields, lets check this is a CXX record and check the base classes
    const CXXRecordDecl *cxx_record_decl = dyn_cast<CXXRecordDecl>(record_decl);
    if (cxx_record_decl)
    {
        CXXRecordDecl::base_class_const_iterator base_class, base_class_end;
        for (base_class = cxx_record_decl->bases_begin(), base_class_end = cxx_record_decl->bases_end();
             base_class != base_class_end;
             ++base_class)
        {
            const CXXRecordDecl *base_class_decl = cast<CXXRecordDecl>(base_class->getType()->getAs<RecordType>()->getDecl());
            if (RecordHasFields(base_class_decl))
                return true;
        }
    }
    return false;
}

#pragma mark Objective C Classes

CompilerType
ClangASTContext::CreateObjCClass
(
    const char *name, 
    DeclContext *decl_ctx, 
    bool isForwardDecl, 
    bool isInternal,
    ClangASTMetadata *metadata
)
{
    ASTContext *ast = getASTContext();
    assert (ast != nullptr);
    assert (name && name[0]);
    if (decl_ctx == nullptr)
        decl_ctx = ast->getTranslationUnitDecl();

    ObjCInterfaceDecl *decl = ObjCInterfaceDecl::Create (*ast,
                                                         decl_ctx,
                                                         SourceLocation(),
                                                         &ast->Idents.get(name),
                                                         nullptr,
                                                         nullptr,
                                                         SourceLocation(),
                                                         /*isForwardDecl,*/
                                                         isInternal);
    
    if (decl && metadata)
        SetMetadata(ast, decl, *metadata);
    
    return CompilerType (ast, ast->getObjCInterfaceType(decl));
}

static inline bool
BaseSpecifierIsEmpty (const CXXBaseSpecifier *b)
{
    return ClangASTContext::RecordHasFields(b->getType()->getAsCXXRecordDecl()) == false;
}

uint32_t
ClangASTContext::GetNumBaseClasses (const CXXRecordDecl *cxx_record_decl, bool omit_empty_base_classes)
{
    uint32_t num_bases = 0;
    if (cxx_record_decl)
    {
        if (omit_empty_base_classes)
        {
            CXXRecordDecl::base_class_const_iterator base_class, base_class_end;
            for (base_class = cxx_record_decl->bases_begin(), base_class_end = cxx_record_decl->bases_end();
                 base_class != base_class_end;
                 ++base_class)
            {
                // Skip empty base classes
                if (omit_empty_base_classes)
                {
                    if (BaseSpecifierIsEmpty (base_class))
                        continue;
                }
                ++num_bases;
            }
        }
        else
            num_bases = cxx_record_decl->getNumBases();
    }
    return num_bases;
}


#pragma mark Namespace Declarations

NamespaceDecl *
ClangASTContext::GetUniqueNamespaceDeclaration (const char *name, DeclContext *decl_ctx)
{
    NamespaceDecl *namespace_decl = nullptr;
    ASTContext *ast = getASTContext();
    TranslationUnitDecl *translation_unit_decl = ast->getTranslationUnitDecl ();
    if (decl_ctx == nullptr)
        decl_ctx = translation_unit_decl;
    
    if (name)
    {
        IdentifierInfo &identifier_info = ast->Idents.get(name);
        DeclarationName decl_name (&identifier_info);
        clang::DeclContext::lookup_result result = decl_ctx->lookup(decl_name);
        for (NamedDecl *decl : result)
        {
            namespace_decl = dyn_cast<clang::NamespaceDecl>(decl);
            if (namespace_decl)
                return namespace_decl;
        }

        namespace_decl = NamespaceDecl::Create(*ast, 
                                               decl_ctx, 
                                               false, 
                                               SourceLocation(), 
                                               SourceLocation(),
                                               &identifier_info,
                                               nullptr);
        
        decl_ctx->addDecl (namespace_decl);        
    }
    else
    {
        if (decl_ctx == translation_unit_decl)
        {
            namespace_decl = translation_unit_decl->getAnonymousNamespace();
            if (namespace_decl)
                return namespace_decl;
            
            namespace_decl = NamespaceDecl::Create(*ast, 
                                                   decl_ctx,
                                                   false,
                                                   SourceLocation(),
                                                   SourceLocation(),
                                                   nullptr,
                                                   nullptr);
            translation_unit_decl->setAnonymousNamespace (namespace_decl);
            translation_unit_decl->addDecl (namespace_decl);
            assert (namespace_decl == translation_unit_decl->getAnonymousNamespace());
        }
        else
        {
            NamespaceDecl *parent_namespace_decl = cast<NamespaceDecl>(decl_ctx);
            if (parent_namespace_decl)
            {
                namespace_decl = parent_namespace_decl->getAnonymousNamespace();
                if (namespace_decl)
                    return namespace_decl;
                namespace_decl = NamespaceDecl::Create(*ast, 
                                                       decl_ctx, 
                                                       false,
                                                       SourceLocation(), 
                                                       SourceLocation(), 
                                                       nullptr,
                                                       nullptr);
                parent_namespace_decl->setAnonymousNamespace (namespace_decl);
                parent_namespace_decl->addDecl (namespace_decl);
                assert (namespace_decl == parent_namespace_decl->getAnonymousNamespace());
            }
            else
            {
                // BAD!!!
            }
        }
    }
#ifdef LLDB_CONFIGURATION_DEBUG
    VerifyDecl(namespace_decl);
#endif
    return namespace_decl;
}


clang::BlockDecl *
ClangASTContext::CreateBlockDeclaration (clang::DeclContext *ctx)
{
    if (ctx != nullptr)
    {
        clang::BlockDecl *decl = clang::BlockDecl::Create(*getASTContext(), ctx, clang::SourceLocation());
        ctx->addDecl(decl);
        return decl;
    }
    return nullptr;
}

clang::DeclContext *
FindLCABetweenDecls(clang::DeclContext *left, clang::DeclContext *right, clang::DeclContext *root)
{
    if (root == nullptr)
        return nullptr;

    std::set<clang::DeclContext *> path_left;
    for (clang::DeclContext *d = left; d != nullptr; d = d->getParent())
        path_left.insert(d);

    for (clang::DeclContext *d = right; d != nullptr; d = d->getParent())
        if (path_left.find(d) != path_left.end())
            return d;

    return nullptr;
}

clang::UsingDirectiveDecl *
ClangASTContext::CreateUsingDirectiveDeclaration (clang::DeclContext *decl_ctx, clang::NamespaceDecl *ns_decl)
{
    if (decl_ctx != nullptr && ns_decl != nullptr)
    {
        clang::TranslationUnitDecl *translation_unit = (clang::TranslationUnitDecl *)GetTranslationUnitDecl(getASTContext());
        clang::UsingDirectiveDecl *using_decl = clang::UsingDirectiveDecl::Create(*getASTContext(),
            decl_ctx,
            clang::SourceLocation(),
            clang::SourceLocation(),
            clang::NestedNameSpecifierLoc(),
            clang::SourceLocation(),
            ns_decl,
            FindLCABetweenDecls(decl_ctx, ns_decl, translation_unit));
        decl_ctx->addDecl(using_decl);
        return using_decl;
    }
    return nullptr;
}

clang::UsingDecl *
ClangASTContext::CreateUsingDeclaration (clang::DeclContext *current_decl_ctx, clang::NamedDecl *target)
{
    if (current_decl_ctx != nullptr && target != nullptr)
    {
        clang::UsingDecl *using_decl = clang::UsingDecl::Create(*getASTContext(),
            current_decl_ctx,
            clang::SourceLocation(),
            clang::NestedNameSpecifierLoc(),
            clang::DeclarationNameInfo(),
            false);
        clang::UsingShadowDecl *shadow_decl = clang::UsingShadowDecl::Create(*getASTContext(),
            current_decl_ctx,
            clang::SourceLocation(),
            using_decl,
            target);
        using_decl->addShadowDecl(shadow_decl);
        current_decl_ctx->addDecl(using_decl);
        return using_decl;
    }
    return nullptr;
}

clang::VarDecl *
ClangASTContext::CreateVariableDeclaration (clang::DeclContext *decl_context, const char *name, clang::QualType type)
{
    if (decl_context != nullptr)
    {
        clang::VarDecl *var_decl = clang::VarDecl::Create(*getASTContext(),
            decl_context,
            clang::SourceLocation(),
            clang::SourceLocation(),
            name && name[0] ? &getASTContext()->Idents.getOwn(name) : nullptr,
            type,
            nullptr,
            clang::SC_None);
        var_decl->setAccess(clang::AS_public);
        decl_context->addDecl(var_decl);
        return var_decl;
    }
    return nullptr;
}

#pragma mark Function Types

FunctionDecl *
ClangASTContext::CreateFunctionDeclaration (DeclContext *decl_ctx,
                                            const char *name,
                                            const CompilerType &function_clang_type,
                                            int storage,
                                            bool is_inline)
{
    FunctionDecl *func_decl = nullptr;
    ASTContext *ast = getASTContext();
    if (decl_ctx == nullptr)
        decl_ctx = ast->getTranslationUnitDecl();

    
    const bool hasWrittenPrototype = true;
    const bool isConstexprSpecified = false;

    if (name && name[0])
    {
        func_decl = FunctionDecl::Create (*ast,
                                          decl_ctx,
                                          SourceLocation(),
                                          SourceLocation(),
                                          DeclarationName (&ast->Idents.get(name)),
                                          GetQualType(function_clang_type),
                                          nullptr,
                                          (clang::StorageClass)storage,
                                          is_inline,
                                          hasWrittenPrototype,
                                          isConstexprSpecified);
    }
    else
    {
        func_decl = FunctionDecl::Create (*ast,
                                          decl_ctx,
                                          SourceLocation(),
                                          SourceLocation(),
                                          DeclarationName (),
                                          GetQualType(function_clang_type),
                                          nullptr,
                                          (clang::StorageClass)storage,
                                          is_inline,
                                          hasWrittenPrototype,
                                          isConstexprSpecified);
    }
    if (func_decl)
        decl_ctx->addDecl (func_decl);
    
#ifdef LLDB_CONFIGURATION_DEBUG
    VerifyDecl(func_decl);
#endif
    
    return func_decl;
}

CompilerType
ClangASTContext::CreateFunctionType (ASTContext *ast,
                                     const CompilerType& result_type,
                                     const CompilerType *args,
                                     unsigned num_args, 
                                     bool is_variadic, 
                                     unsigned type_quals)
{
    assert (ast != nullptr);
    std::vector<QualType> qual_type_args;
    for (unsigned i=0; i<num_args; ++i)
        qual_type_args.push_back (GetQualType(args[i]));

    // TODO: Detect calling convention in DWARF?
    FunctionProtoType::ExtProtoInfo proto_info;
    proto_info.Variadic = is_variadic;
    proto_info.ExceptionSpec = EST_None;
    proto_info.TypeQuals = type_quals;
    proto_info.RefQualifier = RQ_None;

    return CompilerType (ast, ast->getFunctionType (GetQualType(result_type),
                                                    qual_type_args,
                                                    proto_info));
}

ParmVarDecl *
ClangASTContext::CreateParameterDeclaration (const char *name, const CompilerType &param_type, int storage)
{
    ASTContext *ast = getASTContext();
    assert (ast != nullptr);
    return ParmVarDecl::Create(*ast,
                                ast->getTranslationUnitDecl(),
                                SourceLocation(),
                                SourceLocation(),
                                name && name[0] ? &ast->Idents.get(name) : nullptr,
                                GetQualType(param_type),
                                nullptr,
                                (clang::StorageClass)storage,
                                nullptr);
}

void
ClangASTContext::SetFunctionParameters (FunctionDecl *function_decl, ParmVarDecl **params, unsigned num_params)
{
    if (function_decl)
        function_decl->setParams (ArrayRef<ParmVarDecl*>(params, num_params));
}


#pragma mark Array Types

CompilerType
ClangASTContext::CreateArrayType (const CompilerType &element_type,
                                  size_t element_count,
                                  bool is_vector)
{
    if (element_type.IsValid())
    {
        ASTContext *ast = getASTContext();
        assert (ast != nullptr);

        if (is_vector)
        {
            return CompilerType (ast, ast->getExtVectorType(GetQualType(element_type), element_count));
        }
        else
        {
        
            llvm::APInt ap_element_count (64, element_count);
            if (element_count == 0)
            {
                return CompilerType (ast, ast->getIncompleteArrayType (GetQualType(element_type),
                                                                       ArrayType::Normal,
                                                                       0));
            }
            else
            {
                return CompilerType (ast, ast->getConstantArrayType (GetQualType(element_type),
                                                                     ap_element_count,
                                                                     ArrayType::Normal,
                                                                     0));
            }
        }
    }
    return CompilerType();
}

CompilerType
ClangASTContext::GetOrCreateStructForIdentifier (const ConstString &type_name,
                                                 const std::initializer_list< std::pair < const char *, CompilerType > >& type_fields,
                                                 bool packed)
{
    CompilerType type;
    if ((type = GetTypeForIdentifier<clang::CXXRecordDecl>(type_name)).IsValid())
        return type;
    type = CreateRecordType(nullptr, lldb::eAccessPublic, type_name.GetCString(), clang::TTK_Struct, lldb::eLanguageTypeC);
    StartTagDeclarationDefinition(type);
    for (const auto& field : type_fields)
        AddFieldToRecordType(type, field.first, field.second, lldb::eAccessPublic, 0);
    if (packed)
        SetIsPacked(type);
    CompleteTagDeclarationDefinition(type);
    return type;
}

#pragma mark Enumeration Types

CompilerType
ClangASTContext::CreateEnumerationType
(
 const char *name,
 DeclContext *decl_ctx,
 const Declaration &decl,
 const CompilerType &integer_clang_type
 )
{
    // TODO: Do something intelligent with the Declaration object passed in
    // like maybe filling in the SourceLocation with it...
    ASTContext *ast = getASTContext();
    
    // TODO: ask about these...
    //    const bool IsScoped = false;
    //    const bool IsFixed = false;
    
    EnumDecl *enum_decl = EnumDecl::Create (*ast,
                                            decl_ctx,
                                            SourceLocation(),
                                            SourceLocation(),
                                            name && name[0] ? &ast->Idents.get(name) : nullptr,
                                            nullptr,
                                            false,  // IsScoped
                                            false,  // IsScopedUsingClassTag
                                            false); // IsFixed
    
    
    if (enum_decl)
    {
        // TODO: check if we should be setting the promotion type too?
        enum_decl->setIntegerType(GetQualType(integer_clang_type));
        
        enum_decl->setAccess(AS_public); // TODO respect what's in the debug info
        
        return CompilerType (ast, ast->getTagDeclType(enum_decl));
    }
    return CompilerType();
}

// Disable this for now since I can't seem to get a nicely formatted float
// out of the APFloat class without just getting the float, double or quad
// and then using a formatted print on it which defeats the purpose. We ideally
// would like to get perfect string values for any kind of float semantics
// so we can support remote targets. The code below also requires a patch to
// llvm::APInt.
//bool
//ClangASTContext::ConvertFloatValueToString (ASTContext *ast, lldb::opaque_compiler_type_t clang_type, const uint8_t* bytes, size_t byte_size, int apint_byte_order, std::string &float_str)
//{
//  uint32_t count = 0;
//  bool is_complex = false;
//  if (ClangASTContext::IsFloatingPointType (clang_type, count, is_complex))
//  {
//      unsigned num_bytes_per_float = byte_size / count;
//      unsigned num_bits_per_float = num_bytes_per_float * 8;
//
//      float_str.clear();
//      uint32_t i;
//      for (i=0; i<count; i++)
//      {
//          APInt ap_int(num_bits_per_float, bytes + i * num_bytes_per_float, (APInt::ByteOrder)apint_byte_order);
//          bool is_ieee = false;
//          APFloat ap_float(ap_int, is_ieee);
//          char s[1024];
//          unsigned int hex_digits = 0;
//          bool upper_case = false;
//
//          if (ap_float.convertToHexString(s, hex_digits, upper_case, APFloat::rmNearestTiesToEven) > 0)
//          {
//              if (i > 0)
//                  float_str.append(", ");
//              float_str.append(s);
//              if (i == 1 && is_complex)
//                  float_str.append(1, 'i');
//          }
//      }
//      return !float_str.empty();
//  }
//  return false;
//}

CompilerType
ClangASTContext::GetIntTypeFromBitSize (clang::ASTContext *ast,
                                        size_t bit_size, bool is_signed)
{
    if (ast)
    {
        if (is_signed)
        {
            if (bit_size == ast->getTypeSize(ast->SignedCharTy))
                return CompilerType(ast, ast->SignedCharTy);
            
            if (bit_size == ast->getTypeSize(ast->ShortTy))
                return CompilerType(ast, ast->ShortTy);
            
            if (bit_size == ast->getTypeSize(ast->IntTy))
                return CompilerType(ast, ast->IntTy);
            
            if (bit_size == ast->getTypeSize(ast->LongTy))
                return CompilerType(ast, ast->LongTy);
            
            if (bit_size == ast->getTypeSize(ast->LongLongTy))
                return CompilerType(ast, ast->LongLongTy);
            
            if (bit_size == ast->getTypeSize(ast->Int128Ty))
                return CompilerType(ast, ast->Int128Ty);
        }
        else
        {
            if (bit_size == ast->getTypeSize(ast->UnsignedCharTy))
                return CompilerType(ast, ast->UnsignedCharTy);
            
            if (bit_size == ast->getTypeSize(ast->UnsignedShortTy))
                return CompilerType(ast, ast->UnsignedShortTy);
            
            if (bit_size == ast->getTypeSize(ast->UnsignedIntTy))
                return CompilerType(ast, ast->UnsignedIntTy);
            
            if (bit_size == ast->getTypeSize(ast->UnsignedLongTy))
                return CompilerType(ast, ast->UnsignedLongTy);
            
            if (bit_size == ast->getTypeSize(ast->UnsignedLongLongTy))
                return CompilerType(ast, ast->UnsignedLongLongTy);
            
            if (bit_size == ast->getTypeSize(ast->UnsignedInt128Ty))
                return CompilerType(ast, ast->UnsignedInt128Ty);
        }
    }
    return CompilerType();
}

CompilerType
ClangASTContext::GetPointerSizedIntType (clang::ASTContext *ast, bool is_signed)
{
    if (ast)
        return GetIntTypeFromBitSize(ast, ast->getTypeSize(ast->VoidPtrTy), is_signed);
    return CompilerType();
}

void
ClangASTContext::DumpDeclContextHiearchy (clang::DeclContext *decl_ctx)
{
    if (decl_ctx)
    {
        DumpDeclContextHiearchy (decl_ctx->getParent());

        clang::NamedDecl *named_decl = llvm::dyn_cast<clang::NamedDecl>(decl_ctx);
        if (named_decl)
        {
            printf ("%20s: %s\n", decl_ctx->getDeclKindName(), named_decl->getDeclName().getAsString().c_str());
        }
        else
        {
            printf ("%20s\n", decl_ctx->getDeclKindName());
        }
    }
}

void
ClangASTContext::DumpDeclHiearchy (clang::Decl *decl)
{
    if (decl == nullptr)
        return;
    DumpDeclContextHiearchy(decl->getDeclContext());

    clang::RecordDecl *record_decl = llvm::dyn_cast<clang::RecordDecl>(decl);
    if (record_decl)
    {
        printf ("%20s: %s%s\n", decl->getDeclKindName(), record_decl->getDeclName().getAsString().c_str(), record_decl->isInjectedClassName() ? " (injected class name)" : "");

    }
    else
    {
        clang::NamedDecl *named_decl = llvm::dyn_cast<clang::NamedDecl>(decl);
        if (named_decl)
        {
            printf ("%20s: %s\n", decl->getDeclKindName(), named_decl->getDeclName().getAsString().c_str());
        }
        else
        {
            printf ("%20s\n", decl->getDeclKindName());
        }
    }
}

bool
ClangASTContext::DeclsAreEquivalent (clang::Decl *lhs_decl, clang::Decl *rhs_decl)
{
    if (lhs_decl && rhs_decl)
    {
        //----------------------------------------------------------------------
        // Make sure the decl kinds match first
        //----------------------------------------------------------------------
        const clang::Decl::Kind lhs_decl_kind = lhs_decl->getKind();
        const clang::Decl::Kind rhs_decl_kind = rhs_decl->getKind();

        if (lhs_decl_kind == rhs_decl_kind)
        {
            //------------------------------------------------------------------
            // Now check that the decl contexts kinds are all equivalent
            // before we have to check any names of the decl contexts...
            //------------------------------------------------------------------
            clang::DeclContext *lhs_decl_ctx = lhs_decl->getDeclContext();
            clang::DeclContext *rhs_decl_ctx = rhs_decl->getDeclContext();
            if (lhs_decl_ctx && rhs_decl_ctx)
            {
                while (1)
                {
                    if (lhs_decl_ctx && rhs_decl_ctx)
                    {
                        const clang::Decl::Kind lhs_decl_ctx_kind = lhs_decl_ctx->getDeclKind();
                        const clang::Decl::Kind rhs_decl_ctx_kind = rhs_decl_ctx->getDeclKind();
                        if (lhs_decl_ctx_kind == rhs_decl_ctx_kind)
                        {
                            lhs_decl_ctx = lhs_decl_ctx->getParent();
                            rhs_decl_ctx = rhs_decl_ctx->getParent();

                            if (lhs_decl_ctx == nullptr && rhs_decl_ctx == nullptr)
                                break;
                        }
                        else
                            return false;
                    }
                    else
                        return false;
                }

                //--------------------------------------------------------------
                // Now make sure the name of the decls match
                //--------------------------------------------------------------
                clang::NamedDecl *lhs_named_decl = llvm::dyn_cast<clang::NamedDecl>(lhs_decl);
                clang::NamedDecl *rhs_named_decl = llvm::dyn_cast<clang::NamedDecl>(rhs_decl);
                if (lhs_named_decl && rhs_named_decl)
                {
                    clang::DeclarationName lhs_decl_name = lhs_named_decl->getDeclName();
                    clang::DeclarationName rhs_decl_name = rhs_named_decl->getDeclName();
                    if (lhs_decl_name.getNameKind() == rhs_decl_name.getNameKind())
                    {
                        if (lhs_decl_name.getAsString() != rhs_decl_name.getAsString())
                            return false;
                    }
                    else
                        return false;
                }
                else
                    return false;

                //--------------------------------------------------------------
                // We know that the decl context kinds all match, so now we need
                // to make sure the names match as well
                //--------------------------------------------------------------
                lhs_decl_ctx = lhs_decl->getDeclContext();
                rhs_decl_ctx = rhs_decl->getDeclContext();
                while (1)
                {
                    switch (lhs_decl_ctx->getDeclKind())
                    {
                    case clang::Decl::TranslationUnit:
                        // We don't care about the translation unit names
                        return true;
                    default:
                        {
                            clang::NamedDecl *lhs_named_decl = llvm::dyn_cast<clang::NamedDecl>(lhs_decl_ctx);
                            clang::NamedDecl *rhs_named_decl = llvm::dyn_cast<clang::NamedDecl>(rhs_decl_ctx);
                            if (lhs_named_decl && rhs_named_decl)
                            {
                                clang::DeclarationName lhs_decl_name = lhs_named_decl->getDeclName();
                                clang::DeclarationName rhs_decl_name = rhs_named_decl->getDeclName();
                                if (lhs_decl_name.getNameKind() == rhs_decl_name.getNameKind())
                                {
                                    if (lhs_decl_name.getAsString() != rhs_decl_name.getAsString())
                                        return false;
                                }
                                else
                                    return false;
                            }
                            else
                                return false;
                        }
                        break;

                    }
                    lhs_decl_ctx = lhs_decl_ctx->getParent();
                    rhs_decl_ctx = rhs_decl_ctx->getParent();
                }
            }
        }
    }
    return false;
}
bool
ClangASTContext::GetCompleteDecl (clang::ASTContext *ast,
                                  clang::Decl *decl)
{
    if (!decl)
        return false;
    
    ExternalASTSource *ast_source = ast->getExternalSource();
    
    if (!ast_source)
        return false;
        
    if (clang::TagDecl *tag_decl = llvm::dyn_cast<clang::TagDecl>(decl))
    {
        if (tag_decl->isCompleteDefinition())
            return true;
        
        if (!tag_decl->hasExternalLexicalStorage())
            return false;
        
        ast_source->CompleteType(tag_decl);
        
        return !tag_decl->getTypeForDecl()->isIncompleteType();
    }
    else if (clang::ObjCInterfaceDecl *objc_interface_decl = llvm::dyn_cast<clang::ObjCInterfaceDecl>(decl))
    {
        if (objc_interface_decl->getDefinition())
            return true;
        
        if (!objc_interface_decl->hasExternalLexicalStorage())
            return false;
        
        ast_source->CompleteType(objc_interface_decl);
        
        return !objc_interface_decl->getTypeForDecl()->isIncompleteType();
    }
    else
    {
        return false;
    }
}

void
ClangASTContext::SetMetadataAsUserID (const void *object,
                                      user_id_t user_id)
{
    ClangASTMetadata meta_data;
    meta_data.SetUserID (user_id);
    SetMetadata (object, meta_data);
}

void
ClangASTContext::SetMetadata (clang::ASTContext *ast,
                              const void *object,
                              ClangASTMetadata &metadata)
{
    ClangExternalASTSourceCommon *external_source =
        ClangExternalASTSourceCommon::Lookup(ast->getExternalSource());
    
    if (external_source)
        external_source->SetMetadata(object, metadata);
}

ClangASTMetadata *
ClangASTContext::GetMetadata (clang::ASTContext *ast,
                              const void *object)
{
    ClangExternalASTSourceCommon *external_source =
        ClangExternalASTSourceCommon::Lookup(ast->getExternalSource());
    
    if (external_source && external_source->HasMetadata(object))
        return external_source->GetMetadata(object);
    else
        return nullptr;
}

clang::DeclContext *
ClangASTContext::GetAsDeclContext (clang::CXXMethodDecl *cxx_method_decl)
{
    return llvm::dyn_cast<clang::DeclContext>(cxx_method_decl);
}

clang::DeclContext *
ClangASTContext::GetAsDeclContext (clang::ObjCMethodDecl *objc_method_decl)
{
    return llvm::dyn_cast<clang::DeclContext>(objc_method_decl);
}

bool
ClangASTContext::SetTagTypeKind (clang::QualType tag_qual_type, int kind) const
{
    const clang::Type *clang_type = tag_qual_type.getTypePtr();
    if (clang_type)
    {
        const clang::TagType *tag_type = llvm::dyn_cast<clang::TagType>(clang_type);
        if (tag_type)
        {
            clang::TagDecl *tag_decl = llvm::dyn_cast<clang::TagDecl>(tag_type->getDecl());
            if (tag_decl)
            {
                tag_decl->setTagKind ((clang::TagDecl::TagKind)kind);
                return true;
            }
        }
    }
    return false;
}


bool
ClangASTContext::SetDefaultAccessForRecordFields (clang::RecordDecl* record_decl,
                                                  int default_accessibility,
                                                  int *assigned_accessibilities,
                                                  size_t num_assigned_accessibilities)
{
    if (record_decl)
    {
        uint32_t field_idx;
        clang::RecordDecl::field_iterator field, field_end;
        for (field = record_decl->field_begin(), field_end = record_decl->field_end(), field_idx = 0;
             field != field_end;
             ++field, ++field_idx)
        {
            // If no accessibility was assigned, assign the correct one
            if (field_idx < num_assigned_accessibilities && assigned_accessibilities[field_idx] == clang::AS_none)
                field->setAccess ((clang::AccessSpecifier)default_accessibility);
        }
        return true;
    }
    return false;
}

clang::DeclContext *
ClangASTContext::GetDeclContextForType (const CompilerType& type)
{
    return GetDeclContextForType(GetQualType(type));
}

clang::DeclContext *
ClangASTContext::GetDeclContextForType (clang::QualType type)
{
    if (type.isNull())
        return nullptr;
    
    clang::QualType qual_type = type.getCanonicalType();
    const clang::Type::TypeClass type_class = qual_type->getTypeClass();
    switch (type_class)
    {
        case clang::Type::ObjCInterface:            return llvm::cast<clang::ObjCObjectType>(qual_type.getTypePtr())->getInterface();
        case clang::Type::ObjCObjectPointer:        return GetDeclContextForType (llvm::cast<clang::ObjCObjectPointerType>(qual_type.getTypePtr())->getPointeeType());
        case clang::Type::Record:                   return llvm::cast<clang::RecordType>(qual_type)->getDecl();
        case clang::Type::Enum:                     return llvm::cast<clang::EnumType>(qual_type)->getDecl();
        case clang::Type::Typedef:                  return GetDeclContextForType (llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType());
        case clang::Type::Auto:                     return GetDeclContextForType (llvm::cast<clang::AutoType>(qual_type)->getDeducedType());
        case clang::Type::Elaborated:               return GetDeclContextForType (llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType());
        case clang::Type::Paren:                    return GetDeclContextForType (llvm::cast<clang::ParenType>(qual_type)->desugar());
        default:
            break;
    }
    // No DeclContext in this type...
    return nullptr;
}

static bool
GetCompleteQualType (clang::ASTContext *ast, clang::QualType qual_type, bool allow_completion = true)
{
    const clang::Type::TypeClass type_class = qual_type->getTypeClass();
    switch (type_class)
    {
        case clang::Type::ConstantArray:
        case clang::Type::IncompleteArray:
        case clang::Type::VariableArray:
            {
                const clang::ArrayType *array_type = llvm::dyn_cast<clang::ArrayType>(qual_type.getTypePtr());
                
                if (array_type)
                    return GetCompleteQualType (ast, array_type->getElementType(), allow_completion);
            }
            break;
        case clang::Type::Record:
            {
                clang::CXXRecordDecl *cxx_record_decl = qual_type->getAsCXXRecordDecl();
                if (cxx_record_decl)
                {
                    if (cxx_record_decl->hasExternalLexicalStorage())
                    {
                        const bool is_complete = cxx_record_decl->isCompleteDefinition();
                        const bool fields_loaded = cxx_record_decl->hasLoadedFieldsFromExternalStorage();
                        if (is_complete && fields_loaded)
                            return true;

                        if (!allow_completion)
                            return false;

                        // Call the field_begin() accessor to for it to use the external source
                        // to load the fields...
                        clang::ExternalASTSource *external_ast_source = ast->getExternalSource();
                        if (external_ast_source)
                        {
                            external_ast_source->CompleteType(cxx_record_decl);
                            if (cxx_record_decl->isCompleteDefinition())
                            {
                                cxx_record_decl->setHasLoadedFieldsFromExternalStorage (true);
                                cxx_record_decl->field_begin();
                            }
                        }
                    }
                }
                const clang::TagType *tag_type = llvm::cast<clang::TagType>(qual_type.getTypePtr());
                return !tag_type->isIncompleteType();
            }
            break;

        case clang::Type::Enum:
            {
                const clang::TagType *tag_type = llvm::dyn_cast<clang::TagType>(qual_type.getTypePtr());
                if (tag_type)
                {
                    clang::TagDecl *tag_decl = tag_type->getDecl();
                    if (tag_decl)
                    {
                        if (tag_decl->getDefinition())
                            return true;
                        
                        if (!allow_completion)
                            return false;
                        
                        if (tag_decl->hasExternalLexicalStorage())
                        {
                            if (ast)
                            {
                                clang::ExternalASTSource *external_ast_source = ast->getExternalSource();
                                if (external_ast_source)
                                {
                                    external_ast_source->CompleteType(tag_decl);
                                    return !tag_type->isIncompleteType();
                                }
                            }
                        }
                        return false;
                    }
                }
                
            }
            break;
        case clang::Type::ObjCObject:
        case clang::Type::ObjCInterface:
            {
                const clang::ObjCObjectType *objc_class_type = llvm::dyn_cast<clang::ObjCObjectType>(qual_type);
                if (objc_class_type)
                {
                    clang::ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterface();
                    // We currently can't complete objective C types through the newly added ASTContext
                    // because it only supports TagDecl objects right now...
                    if (class_interface_decl)
                    {
                        if (class_interface_decl->getDefinition())
                            return true;
                        
                        if (!allow_completion)
                            return false;
                        
                        if (class_interface_decl->hasExternalLexicalStorage())
                        {
                            if (ast)
                            {
                                clang::ExternalASTSource *external_ast_source = ast->getExternalSource();
                                if (external_ast_source)
                                {
                                    external_ast_source->CompleteType (class_interface_decl);
                                    return !objc_class_type->isIncompleteType();
                                }
                            }
                        }
                        return false;
                    }
                }
            }
            break;
            
        case clang::Type::Typedef:
            return GetCompleteQualType (ast, llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType(), allow_completion);
            
        case clang::Type::Auto:
            return GetCompleteQualType (ast, llvm::cast<clang::AutoType>(qual_type)->getDeducedType(), allow_completion);
            
        case clang::Type::Elaborated:
            return GetCompleteQualType (ast, llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType(), allow_completion);
            
        case clang::Type::Paren:
            return GetCompleteQualType (ast, llvm::cast<clang::ParenType>(qual_type)->desugar(), allow_completion);

        case clang::Type::Attributed:
            return GetCompleteQualType (ast, llvm::cast<clang::AttributedType>(qual_type)->getModifiedType(), allow_completion);

        default:
            break;
    }
    
    return true;
}

static clang::ObjCIvarDecl::AccessControl
ConvertAccessTypeToObjCIvarAccessControl (AccessType access)
{
    switch (access)
    {
        case eAccessNone:      return clang::ObjCIvarDecl::None;
        case eAccessPublic:    return clang::ObjCIvarDecl::Public;
        case eAccessPrivate:   return clang::ObjCIvarDecl::Private;
        case eAccessProtected: return clang::ObjCIvarDecl::Protected;
        case eAccessPackage:   return clang::ObjCIvarDecl::Package;
    }
    return clang::ObjCIvarDecl::None;
}


//----------------------------------------------------------------------
// Tests
//----------------------------------------------------------------------

bool
ClangASTContext::IsAggregateType (lldb::opaque_compiler_type_t type)
{
    clang::QualType qual_type (GetCanonicalQualType(type));
    
    const clang::Type::TypeClass type_class = qual_type->getTypeClass();
    switch (type_class)
    {
        case clang::Type::IncompleteArray:
        case clang::Type::VariableArray:
        case clang::Type::ConstantArray:
        case clang::Type::ExtVector:
        case clang::Type::Vector:
        case clang::Type::Record:
        case clang::Type::ObjCObject:
        case clang::Type::ObjCInterface:
            return true;
        case clang::Type::Auto:
            return IsAggregateType(llvm::cast<clang::AutoType>(qual_type)->getDeducedType().getAsOpaquePtr());
        case clang::Type::Elaborated:
            return IsAggregateType(llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType().getAsOpaquePtr());
        case clang::Type::Typedef:
            return IsAggregateType(llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType().getAsOpaquePtr());
        case clang::Type::Paren:
            return IsAggregateType(llvm::cast<clang::ParenType>(qual_type)->desugar().getAsOpaquePtr());
        default:
            break;
    }
    // The clang type does have a value
    return false;
}

bool
ClangASTContext::IsAnonymousType (lldb::opaque_compiler_type_t type)
{
    clang::QualType qual_type (GetCanonicalQualType(type));
    
    const clang::Type::TypeClass type_class = qual_type->getTypeClass();
    switch (type_class)
    {
        case clang::Type::Record:
        {
            if (const clang::RecordType *record_type = llvm::dyn_cast_or_null<clang::RecordType>(qual_type.getTypePtrOrNull()))
            {
                if (const clang::RecordDecl *record_decl = record_type->getDecl())
                {
                    return record_decl->isAnonymousStructOrUnion();
                }
            }
            break;
        }
        case clang::Type::Auto:
            return IsAnonymousType(llvm::cast<clang::AutoType>(qual_type)->getDeducedType().getAsOpaquePtr());
        case clang::Type::Elaborated:
            return IsAnonymousType(llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType().getAsOpaquePtr());
        case clang::Type::Typedef:
            return IsAnonymousType(llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType().getAsOpaquePtr());
        case clang::Type::Paren:
            return IsAnonymousType(llvm::cast<clang::ParenType>(qual_type)->desugar().getAsOpaquePtr());
        default:
            break;
    }
    // The clang type does have a value
    return false;
}

bool
ClangASTContext::IsArrayType (lldb::opaque_compiler_type_t type,
                              CompilerType *element_type_ptr,
                              uint64_t *size,
                              bool *is_incomplete)
{
    clang::QualType qual_type (GetCanonicalQualType(type));
    
    const clang::Type::TypeClass type_class = qual_type->getTypeClass();
    switch (type_class)
    {
        default:
            break;

        case clang::Type::ConstantArray:
            if (element_type_ptr)
                element_type_ptr->SetCompilerType (getASTContext(), llvm::cast<clang::ConstantArrayType>(qual_type)->getElementType());
            if (size)
                *size = llvm::cast<clang::ConstantArrayType>(qual_type)->getSize().getLimitedValue(ULLONG_MAX);
            if (is_incomplete)
                *is_incomplete = false;
            return true;

        case clang::Type::IncompleteArray:
            if (element_type_ptr)
                element_type_ptr->SetCompilerType (getASTContext(), llvm::cast<clang::IncompleteArrayType>(qual_type)->getElementType());
            if (size)
                *size = 0;
            if (is_incomplete)
                *is_incomplete = true;
            return true;

        case clang::Type::VariableArray:
            if (element_type_ptr)
                element_type_ptr->SetCompilerType (getASTContext(), llvm::cast<clang::VariableArrayType>(qual_type)->getElementType());
            if (size)
                *size = 0;
            if (is_incomplete)
                *is_incomplete = false;
            return true;

        case clang::Type::DependentSizedArray:
            if (element_type_ptr)
                element_type_ptr->SetCompilerType (getASTContext(), llvm::cast<clang::DependentSizedArrayType>(qual_type)->getElementType());
            if (size)
                *size = 0;
            if (is_incomplete)
                *is_incomplete = false;
            return true;

        case clang::Type::Typedef:
            return IsArrayType(llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType().getAsOpaquePtr(),
                               element_type_ptr,
                               size,
                               is_incomplete);
        case clang::Type::Auto:
            return IsArrayType(llvm::cast<clang::AutoType>(qual_type)->getDeducedType().getAsOpaquePtr(),
                               element_type_ptr,
                               size,
                               is_incomplete);
        case clang::Type::Elaborated:
            return IsArrayType(llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType().getAsOpaquePtr(),
                               element_type_ptr,
                               size,
                               is_incomplete);
        case clang::Type::Paren:
            return IsArrayType(llvm::cast<clang::ParenType>(qual_type)->desugar().getAsOpaquePtr(),
                               element_type_ptr,
                               size,
                               is_incomplete);
    }
    if (element_type_ptr)
        element_type_ptr->Clear();
    if (size)
        *size = 0;
    if (is_incomplete)
        *is_incomplete = false;
    return false;
}

bool
ClangASTContext::IsVectorType (lldb::opaque_compiler_type_t type,
                               CompilerType *element_type,
                               uint64_t *size)
{
    clang::QualType qual_type (GetCanonicalQualType(type));
    
    const clang::Type::TypeClass type_class = qual_type->getTypeClass();
    switch (type_class)
    {
        case clang::Type::Vector:
        {
            const clang::VectorType *vector_type = qual_type->getAs<clang::VectorType>();
            if (vector_type)
            {
                if (size)
                    *size = vector_type->getNumElements();
                if (element_type)
                    *element_type = CompilerType(getASTContext(), vector_type->getElementType());
            }
            return true;
        }
            break;
        case clang::Type::ExtVector:
        {
            const clang::ExtVectorType *ext_vector_type = qual_type->getAs<clang::ExtVectorType>();
            if (ext_vector_type)
            {
                if (size)
                    *size = ext_vector_type->getNumElements();
                if (element_type)
                    *element_type = CompilerType(getASTContext(), ext_vector_type->getElementType());
            }
            return true;
        }
        default:
            break;
    }
    return false;
}

bool
ClangASTContext::IsRuntimeGeneratedType (lldb::opaque_compiler_type_t type)
{
    clang::DeclContext* decl_ctx = ClangASTContext::GetASTContext(getASTContext())->GetDeclContextForType(GetQualType(type));
    if (!decl_ctx)
        return false;
    
    if (!llvm::isa<clang::ObjCInterfaceDecl>(decl_ctx))
        return false;
    
    clang::ObjCInterfaceDecl *result_iface_decl = llvm::dyn_cast<clang::ObjCInterfaceDecl>(decl_ctx);
    
    ClangASTMetadata* ast_metadata = ClangASTContext::GetMetadata(getASTContext(), result_iface_decl);
    if (!ast_metadata)
        return false;
    return (ast_metadata->GetISAPtr() != 0);
}

bool
ClangASTContext::IsCharType (lldb::opaque_compiler_type_t type)
{
    return GetQualType(type).getUnqualifiedType()->isCharType();
}


bool
ClangASTContext::IsCompleteType (lldb::opaque_compiler_type_t type)
{
    const bool allow_completion = false;
    return GetCompleteQualType (getASTContext(), GetQualType(type), allow_completion);
}

bool
ClangASTContext::IsConst(lldb::opaque_compiler_type_t type)
{
    return GetQualType(type).isConstQualified();
}

bool
ClangASTContext::IsCStringType (lldb::opaque_compiler_type_t type, uint32_t &length)
{
    CompilerType pointee_or_element_clang_type;
    length = 0;
    Flags type_flags (GetTypeInfo (type, &pointee_or_element_clang_type));
    
    if (!pointee_or_element_clang_type.IsValid())
        return false;
    
    if (type_flags.AnySet (eTypeIsArray | eTypeIsPointer))
    {
        if (pointee_or_element_clang_type.IsCharType())
        {
            if (type_flags.Test (eTypeIsArray))
            {
                // We know the size of the array and it could be a C string
                // since it is an array of characters
                length = llvm::cast<clang::ConstantArrayType>(GetCanonicalQualType(type).getTypePtr())->getSize().getLimitedValue();
            }
            return true;
            
        }
    }
    return false;
}

bool
ClangASTContext::IsFunctionType (lldb::opaque_compiler_type_t type, bool *is_variadic_ptr)
{
    if (type)
    {
        clang::QualType qual_type (GetCanonicalQualType(type));
        
        if (qual_type->isFunctionType())
        {
            if (is_variadic_ptr)
            {
                const clang::FunctionProtoType *function_proto_type = llvm::dyn_cast<clang::FunctionProtoType>(qual_type.getTypePtr());
                if (function_proto_type)
                    *is_variadic_ptr = function_proto_type->isVariadic();
                else
                    *is_variadic_ptr = false;
            }
            return true;
        }
        
        const clang::Type::TypeClass type_class = qual_type->getTypeClass();
        switch (type_class)
        {
            default:
                break;
            case clang::Type::Typedef:
                return IsFunctionType(llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType().getAsOpaquePtr(), nullptr);
            case clang::Type::Auto:
                return IsFunctionType(llvm::cast<clang::AutoType>(qual_type)->getDeducedType().getAsOpaquePtr(), nullptr);
            case clang::Type::Elaborated:
                return IsFunctionType(llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType().getAsOpaquePtr(), nullptr);
            case clang::Type::Paren:
                return IsFunctionType(llvm::cast<clang::ParenType>(qual_type)->desugar().getAsOpaquePtr(), nullptr);
            case clang::Type::LValueReference:
            case clang::Type::RValueReference:
                {
                    const clang::ReferenceType *reference_type = llvm::cast<clang::ReferenceType>(qual_type.getTypePtr());
                    if (reference_type)
                        return IsFunctionType(reference_type->getPointeeType().getAsOpaquePtr(), nullptr);
                }
                break;
        }
    }
    return false;
}

// Used to detect "Homogeneous Floating-point Aggregates"
uint32_t
ClangASTContext::IsHomogeneousAggregate (lldb::opaque_compiler_type_t type, CompilerType* base_type_ptr)
{
    if (!type)
        return 0;
    
    clang::QualType qual_type(GetCanonicalQualType(type));
    const clang::Type::TypeClass type_class = qual_type->getTypeClass();
    switch (type_class)
    {
        case clang::Type::Record:
            if (GetCompleteType (type))
            {
                const clang::CXXRecordDecl *cxx_record_decl = qual_type->getAsCXXRecordDecl();
                if (cxx_record_decl)
                {
                    if (cxx_record_decl->getNumBases() ||
                        cxx_record_decl->isDynamicClass())
                        return 0;
                }
                const clang::RecordType *record_type = llvm::cast<clang::RecordType>(qual_type.getTypePtr());
                if (record_type)
                {
                    const clang::RecordDecl *record_decl = record_type->getDecl();
                    if (record_decl)
                    {
                        // We are looking for a structure that contains only floating point types
                        clang::RecordDecl::field_iterator field_pos, field_end = record_decl->field_end();
                        uint32_t num_fields = 0;
                        bool is_hva = false;
                        bool is_hfa = false;
                        clang::QualType base_qual_type;
                        for (field_pos = record_decl->field_begin(); field_pos != field_end; ++field_pos)
                        {
                            clang::QualType field_qual_type = field_pos->getType();
                            if (field_qual_type->isFloatingType())
                            {
                                if (field_qual_type->isComplexType())
                                    return 0;
                                else
                                {
                                    if (num_fields == 0)
                                        base_qual_type = field_qual_type;
                                    else
                                    {
                                        if (is_hva)
                                            return 0;
                                        is_hfa = true;
                                        if (field_qual_type.getTypePtr() != base_qual_type.getTypePtr())
                                            return 0;
                                    }
                                }
                            }
                            else if (field_qual_type->isVectorType() || field_qual_type->isExtVectorType())
                            {
                                const clang::VectorType *array = field_qual_type.getTypePtr()->getAs<clang::VectorType>();
                                if (array && array->getNumElements() <= 4)
                                {
                                    if (num_fields == 0)
                                        base_qual_type = array->getElementType();
                                    else
                                    {
                                        if (is_hfa)
                                            return 0;
                                        is_hva = true;
                                        if (field_qual_type.getTypePtr() != base_qual_type.getTypePtr())
                                            return 0;
                                    }
                                }
                                else
                                    return 0;
                            }
                            else
                                return 0;
                            ++num_fields;
                        }
                        if (base_type_ptr)
                            *base_type_ptr = CompilerType (getASTContext(), base_qual_type);
                        return num_fields;
                    }
                }
            }
            break;
            
        case clang::Type::Typedef:
            return IsHomogeneousAggregate(llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType().getAsOpaquePtr(), base_type_ptr);

        case clang::Type::Auto:
            return IsHomogeneousAggregate(llvm::cast<clang::AutoType>(qual_type)->getDeducedType().getAsOpaquePtr(), base_type_ptr);
            
        case clang::Type::Elaborated:
            return IsHomogeneousAggregate(llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType().getAsOpaquePtr(), base_type_ptr);
        default:
            break;
    }
    return 0;
}

size_t
ClangASTContext::GetNumberOfFunctionArguments (lldb::opaque_compiler_type_t type)
{
    if (type)
    {
        clang::QualType qual_type (GetCanonicalQualType(type));
        const clang::FunctionProtoType* func = llvm::dyn_cast<clang::FunctionProtoType>(qual_type.getTypePtr());
        if (func)
            return func->getNumParams();
    }
    return 0;
}

CompilerType
ClangASTContext::GetFunctionArgumentAtIndex (lldb::opaque_compiler_type_t type, const size_t index)
{
    if (type)
    {
        clang::QualType qual_type (GetCanonicalQualType(type));
        const clang::FunctionProtoType* func = llvm::dyn_cast<clang::FunctionProtoType>(qual_type.getTypePtr());
        if (func)
        {
            if (index < func->getNumParams())
                return CompilerType(getASTContext(), func->getParamType(index));
        }
    }
    return CompilerType();
}

bool
ClangASTContext::IsFunctionPointerType (lldb::opaque_compiler_type_t type)
{
    if (type)
    {
        clang::QualType qual_type (GetCanonicalQualType(type));
        
        if (qual_type->isFunctionPointerType())
            return true;
        
        const clang::Type::TypeClass type_class = qual_type->getTypeClass();
        switch (type_class)
        {
            default:
                break;
            case clang::Type::Typedef:
                return IsFunctionPointerType (llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType().getAsOpaquePtr());
            case clang::Type::Auto:
                return IsFunctionPointerType (llvm::cast<clang::AutoType>(qual_type)->getDeducedType().getAsOpaquePtr());
            case clang::Type::Elaborated:
                return IsFunctionPointerType (llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType().getAsOpaquePtr());
            case clang::Type::Paren:
                return IsFunctionPointerType (llvm::cast<clang::ParenType>(qual_type)->desugar().getAsOpaquePtr());
                
            case clang::Type::LValueReference:
            case clang::Type::RValueReference:
                {
                    const clang::ReferenceType *reference_type = llvm::cast<clang::ReferenceType>(qual_type.getTypePtr());
                    if (reference_type)
                        return IsFunctionPointerType(reference_type->getPointeeType().getAsOpaquePtr());
                }
                break;
        }
    }
    return false;
    
}

bool
ClangASTContext::IsIntegerType (lldb::opaque_compiler_type_t type, bool &is_signed)
{
    if (!type)
        return false;
    
    clang::QualType qual_type (GetCanonicalQualType(type));
    const clang::BuiltinType *builtin_type = llvm::dyn_cast<clang::BuiltinType>(qual_type->getCanonicalTypeInternal());
    
    if (builtin_type)
    {
        if (builtin_type->isInteger())
        {
            is_signed = builtin_type->isSignedInteger();
            return true;
        }
    }
    
    return false;
}

bool
ClangASTContext::IsPointerType (lldb::opaque_compiler_type_t type, CompilerType *pointee_type)
{
    if (type)
    {
        clang::QualType qual_type (GetCanonicalQualType(type));
        const clang::Type::TypeClass type_class = qual_type->getTypeClass();
        switch (type_class)
        {
            case clang::Type::Builtin:
                switch (llvm::cast<clang::BuiltinType>(qual_type)->getKind())
            {
                default:
                    break;
                case clang::BuiltinType::ObjCId:
                case clang::BuiltinType::ObjCClass:
                    return true;
            }
                return false;
            case clang::Type::ObjCObjectPointer:
                if (pointee_type)
                    pointee_type->SetCompilerType (getASTContext(), llvm::cast<clang::ObjCObjectPointerType>(qual_type)->getPointeeType());
                return true;
            case clang::Type::BlockPointer:
                if (pointee_type)
                    pointee_type->SetCompilerType (getASTContext(), llvm::cast<clang::BlockPointerType>(qual_type)->getPointeeType());
                return true;
            case clang::Type::Pointer:
                if (pointee_type)
                    pointee_type->SetCompilerType (getASTContext(), llvm::cast<clang::PointerType>(qual_type)->getPointeeType());
                return true;
            case clang::Type::MemberPointer:
                if (pointee_type)
                    pointee_type->SetCompilerType (getASTContext(), llvm::cast<clang::MemberPointerType>(qual_type)->getPointeeType());
                return true;
            case clang::Type::Typedef:
                return IsPointerType (llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType().getAsOpaquePtr(), pointee_type);
            case clang::Type::Auto:
                return IsPointerType (llvm::cast<clang::AutoType>(qual_type)->getDeducedType().getAsOpaquePtr(), pointee_type);
            case clang::Type::Elaborated:
                return IsPointerType (llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType().getAsOpaquePtr(), pointee_type);
            case clang::Type::Paren:
                return IsPointerType (llvm::cast<clang::ParenType>(qual_type)->desugar().getAsOpaquePtr(), pointee_type);
            default:
                break;
        }
    }
    if (pointee_type)
        pointee_type->Clear();
    return false;
}


bool
ClangASTContext::IsPointerOrReferenceType (lldb::opaque_compiler_type_t type, CompilerType *pointee_type)
{
    if (type)
    {
        clang::QualType qual_type (GetCanonicalQualType(type));
        const clang::Type::TypeClass type_class = qual_type->getTypeClass();
        switch (type_class)
        {
            case clang::Type::Builtin:
                switch (llvm::cast<clang::BuiltinType>(qual_type)->getKind())
                {
                    default:
                        break;
                    case clang::BuiltinType::ObjCId:
                    case clang::BuiltinType::ObjCClass:
                        return true;
                }
                return false;
            case clang::Type::ObjCObjectPointer:
                if (pointee_type)
                    pointee_type->SetCompilerType(getASTContext(), llvm::cast<clang::ObjCObjectPointerType>(qual_type)->getPointeeType());
                return true;
            case clang::Type::BlockPointer:
                if (pointee_type)
                    pointee_type->SetCompilerType(getASTContext(), llvm::cast<clang::BlockPointerType>(qual_type)->getPointeeType());
                return true;
            case clang::Type::Pointer:
                if (pointee_type)
                    pointee_type->SetCompilerType(getASTContext(), llvm::cast<clang::PointerType>(qual_type)->getPointeeType());
                return true;
            case clang::Type::MemberPointer:
                if (pointee_type)
                    pointee_type->SetCompilerType(getASTContext(), llvm::cast<clang::MemberPointerType>(qual_type)->getPointeeType());
                return true;
            case clang::Type::LValueReference:
                if (pointee_type)
                    pointee_type->SetCompilerType(getASTContext(), llvm::cast<clang::LValueReferenceType>(qual_type)->desugar());
                return true;
            case clang::Type::RValueReference:
                if (pointee_type)
                    pointee_type->SetCompilerType(getASTContext(), llvm::cast<clang::RValueReferenceType>(qual_type)->desugar());
                return true;
            case clang::Type::Typedef:
                return IsPointerOrReferenceType(llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType().getAsOpaquePtr(), pointee_type);
            case clang::Type::Auto:
                return IsPointerOrReferenceType(llvm::cast<clang::AutoType>(qual_type)->getDeducedType().getAsOpaquePtr(), pointee_type);
            case clang::Type::Elaborated:
                return IsPointerOrReferenceType(llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType().getAsOpaquePtr(), pointee_type);
            case clang::Type::Paren:
                return IsPointerOrReferenceType(llvm::cast<clang::ParenType>(qual_type)->desugar().getAsOpaquePtr(), pointee_type);
            default:
                break;
        }
    }
    if (pointee_type)
        pointee_type->Clear();
    return false;
}


bool
ClangASTContext::IsReferenceType (lldb::opaque_compiler_type_t type, CompilerType *pointee_type, bool* is_rvalue)
{
    if (type)
    {
        clang::QualType qual_type (GetCanonicalQualType(type));
        const clang::Type::TypeClass type_class = qual_type->getTypeClass();
        
        switch (type_class)
        {
            case clang::Type::LValueReference:
                if (pointee_type)
                    pointee_type->SetCompilerType(getASTContext(), llvm::cast<clang::LValueReferenceType>(qual_type)->desugar());
                if (is_rvalue)
                    *is_rvalue = false;
                return true;
            case clang::Type::RValueReference:
                if (pointee_type)
                    pointee_type->SetCompilerType(getASTContext(), llvm::cast<clang::RValueReferenceType>(qual_type)->desugar());
                if (is_rvalue)
                    *is_rvalue = true;
                return true;
            case clang::Type::Typedef:
                return IsReferenceType (llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType().getAsOpaquePtr(), pointee_type, is_rvalue);
            case clang::Type::Auto:
                return IsReferenceType (llvm::cast<clang::AutoType>(qual_type)->getDeducedType().getAsOpaquePtr(), pointee_type, is_rvalue);
            case clang::Type::Elaborated:
                return IsReferenceType (llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType().getAsOpaquePtr(), pointee_type, is_rvalue);
            case clang::Type::Paren:
                return IsReferenceType (llvm::cast<clang::ParenType>(qual_type)->desugar().getAsOpaquePtr(), pointee_type, is_rvalue);
                
            default:
                break;
        }
    }
    if (pointee_type)
        pointee_type->Clear();
    return false;
}

bool
ClangASTContext::IsFloatingPointType (lldb::opaque_compiler_type_t type, uint32_t &count, bool &is_complex)
{
    if (type)
    {
        clang::QualType qual_type (GetCanonicalQualType(type));
        
        if (const clang::BuiltinType *BT = llvm::dyn_cast<clang::BuiltinType>(qual_type->getCanonicalTypeInternal()))
        {
            clang::BuiltinType::Kind kind = BT->getKind();
            if (kind >= clang::BuiltinType::Float && kind <= clang::BuiltinType::LongDouble)
            {
                count = 1;
                is_complex = false;
                return true;
            }
        }
        else if (const clang::ComplexType *CT = llvm::dyn_cast<clang::ComplexType>(qual_type->getCanonicalTypeInternal()))
        {
            if (IsFloatingPointType (CT->getElementType().getAsOpaquePtr(), count, is_complex))
            {
                count = 2;
                is_complex = true;
                return true;
            }
        }
        else if (const clang::VectorType *VT = llvm::dyn_cast<clang::VectorType>(qual_type->getCanonicalTypeInternal()))
        {
            if (IsFloatingPointType (VT->getElementType().getAsOpaquePtr(), count, is_complex))
            {
                count = VT->getNumElements();
                is_complex = false;
                return true;
            }
        }
    }
    count = 0;
    is_complex = false;
    return false;
}


bool
ClangASTContext::IsDefined(lldb::opaque_compiler_type_t type)
{
    if (!type)
        return false;
    
    clang::QualType qual_type(GetQualType(type));
    const clang::TagType *tag_type = llvm::dyn_cast<clang::TagType>(qual_type.getTypePtr());
    if (tag_type)
    {
        clang::TagDecl *tag_decl = tag_type->getDecl();
        if (tag_decl)
            return tag_decl->isCompleteDefinition();
        return false;
    }
    else
    {
        const clang::ObjCObjectType *objc_class_type = llvm::dyn_cast<clang::ObjCObjectType>(qual_type);
        if (objc_class_type)
        {
            clang::ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterface();
            if (class_interface_decl)
                return class_interface_decl->getDefinition() != nullptr;
            return false;
        }
    }
    return true;
}

bool
ClangASTContext::IsObjCClassType (const CompilerType& type)
{
    if (type)
    {
        clang::QualType qual_type (GetCanonicalQualType(type));
        
        const clang::ObjCObjectPointerType *obj_pointer_type = llvm::dyn_cast<clang::ObjCObjectPointerType>(qual_type);
        
        if (obj_pointer_type)
            return obj_pointer_type->isObjCClassType();
    }
    return false;
}

bool
ClangASTContext::IsObjCObjectOrInterfaceType (const CompilerType& type)
{
    if (IsClangType(type))
        return GetCanonicalQualType(type)->isObjCObjectOrInterfaceType();
    return false;
}

bool
ClangASTContext::IsPolymorphicClass (lldb::opaque_compiler_type_t type)
{
    if (type)
    {
        clang::QualType qual_type(GetCanonicalQualType(type));
        const clang::Type::TypeClass type_class = qual_type->getTypeClass();
        switch (type_class)
        {
            case clang::Type::Record:
                if (GetCompleteType(type))
                {
                    const clang::RecordType *record_type = llvm::cast<clang::RecordType>(qual_type.getTypePtr());
                    const clang::RecordDecl *record_decl = record_type->getDecl();
                    if (record_decl)
                    {
                        const clang::CXXRecordDecl *cxx_record_decl = llvm::dyn_cast<clang::CXXRecordDecl>(record_decl);
                        if (cxx_record_decl)
                            return cxx_record_decl->isPolymorphic();
                    }
                }
                break;
                
            default:
                break;
        }
    }
    return false;
}

bool
ClangASTContext::IsPossibleDynamicType (lldb::opaque_compiler_type_t type, CompilerType *dynamic_pointee_type,
                                           bool check_cplusplus,
                                           bool check_objc)
{
    clang::QualType pointee_qual_type;
    if (type)
    {
        clang::QualType qual_type (GetCanonicalQualType(type));
        bool success = false;
        const clang::Type::TypeClass type_class = qual_type->getTypeClass();
        switch (type_class)
        {
            case clang::Type::Builtin:
                if (check_objc && llvm::cast<clang::BuiltinType>(qual_type)->getKind() == clang::BuiltinType::ObjCId)
                {
                    if (dynamic_pointee_type)
                        dynamic_pointee_type->SetCompilerType(this, type);
                    return true;
                }
                break;
                
            case clang::Type::ObjCObjectPointer:
                if (check_objc)
                {
                    if (dynamic_pointee_type)
                        dynamic_pointee_type->SetCompilerType(getASTContext(), llvm::cast<clang::ObjCObjectPointerType>(qual_type)->getPointeeType());
                    return true;
                }
                break;
                
            case clang::Type::Pointer:
                pointee_qual_type = llvm::cast<clang::PointerType>(qual_type)->getPointeeType();
                success = true;
                break;
                
            case clang::Type::LValueReference:
            case clang::Type::RValueReference:
                pointee_qual_type = llvm::cast<clang::ReferenceType>(qual_type)->getPointeeType();
                success = true;
                break;
                
            case clang::Type::Typedef:
                return IsPossibleDynamicType (llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType().getAsOpaquePtr(),
                                              dynamic_pointee_type,
                                              check_cplusplus,
                                              check_objc);

            case clang::Type::Auto:
                return IsPossibleDynamicType (llvm::cast<clang::AutoType>(qual_type)->getDeducedType().getAsOpaquePtr(),
                                              dynamic_pointee_type,
                                              check_cplusplus,
                                              check_objc);
                
            case clang::Type::Elaborated:
                return IsPossibleDynamicType (llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType().getAsOpaquePtr(),
                                              dynamic_pointee_type,
                                              check_cplusplus,
                                              check_objc);
                
            case clang::Type::Paren:
                return IsPossibleDynamicType (llvm::cast<clang::ParenType>(qual_type)->desugar().getAsOpaquePtr(),
                                              dynamic_pointee_type,
                                              check_cplusplus,
                                              check_objc);
            default:
                break;
        }
        
        if (success)
        {
            // Check to make sure what we are pointing too is a possible dynamic C++ type
            // We currently accept any "void *" (in case we have a class that has been
            // watered down to an opaque pointer) and virtual C++ classes.
            const clang::Type::TypeClass pointee_type_class = pointee_qual_type.getCanonicalType()->getTypeClass();
            switch (pointee_type_class)
            {
                case clang::Type::Builtin:
                    switch (llvm::cast<clang::BuiltinType>(pointee_qual_type)->getKind())
                {
                    case clang::BuiltinType::UnknownAny:
                    case clang::BuiltinType::Void:
                        if (dynamic_pointee_type)
                            dynamic_pointee_type->SetCompilerType(getASTContext(), pointee_qual_type);
                        return true;
                    default:
                        break;
                }
                    break;
                    
                case clang::Type::Record:
                    if (check_cplusplus)
                    {
                        clang::CXXRecordDecl *cxx_record_decl = pointee_qual_type->getAsCXXRecordDecl();
                        if (cxx_record_decl)
                        {
                            bool is_complete = cxx_record_decl->isCompleteDefinition();
                            
                            if (is_complete)
                                success = cxx_record_decl->isDynamicClass();
                            else
                            {
                                ClangASTMetadata *metadata = ClangASTContext::GetMetadata (getASTContext(), cxx_record_decl);
                                if (metadata)
                                    success = metadata->GetIsDynamicCXXType();
                                else
                                {
                                    is_complete = CompilerType(getASTContext(), pointee_qual_type).GetCompleteType();
                                    if (is_complete)
                                        success = cxx_record_decl->isDynamicClass();
                                    else
                                        success = false;
                                }
                            }
                            
                            if (success)
                            {
                                if (dynamic_pointee_type)
                                    dynamic_pointee_type->SetCompilerType(getASTContext(), pointee_qual_type);
                                return true;
                            }
                        }
                    }
                    break;
                    
                case clang::Type::ObjCObject:
                case clang::Type::ObjCInterface:
                    if (check_objc)
                    {
                        if (dynamic_pointee_type)
                            dynamic_pointee_type->SetCompilerType(getASTContext(), pointee_qual_type);
                        return true;
                    }
                    break;
                    
                default:
                    break;
            }
        }
    }
    if (dynamic_pointee_type)
        dynamic_pointee_type->Clear();
    return false;
}


bool
ClangASTContext::IsScalarType (lldb::opaque_compiler_type_t type)
{
    if (!type)
        return false;
    
    return (GetTypeInfo (type, nullptr) & eTypeIsScalar) != 0;
}

bool
ClangASTContext::IsTypedefType (lldb::opaque_compiler_type_t type)
{
    if (!type)
        return false;
    return GetQualType(type)->getTypeClass() == clang::Type::Typedef;
}

bool
ClangASTContext::IsVoidType (lldb::opaque_compiler_type_t type)
{
    if (!type)
        return false;
    return GetCanonicalQualType(type)->isVoidType();
}

bool
ClangASTContext::SupportsLanguage (lldb::LanguageType language)
{
    return ClangASTContextSupportsLanguage(language);
}

bool
ClangASTContext::GetCXXClassName (const CompilerType& type, std::string &class_name)
{
    if (type)
    {
        clang::QualType qual_type (GetCanonicalQualType(type));
        if (!qual_type.isNull())
        {
            clang::CXXRecordDecl *cxx_record_decl = qual_type->getAsCXXRecordDecl();
            if (cxx_record_decl)
            {
                class_name.assign(cxx_record_decl->getIdentifier()->getNameStart());
                return true;
            }
        }
    }
    class_name.clear();
    return false;
}


bool
ClangASTContext::IsCXXClassType (const CompilerType& type)
{
    if (!type)
        return false;
    
    clang::QualType qual_type (GetCanonicalQualType(type));
    if (!qual_type.isNull() && qual_type->getAsCXXRecordDecl() != nullptr)
        return true;
    return false;
}

bool
ClangASTContext::IsBeingDefined (lldb::opaque_compiler_type_t type)
{
    if (!type)
        return false;
    clang::QualType qual_type (GetCanonicalQualType(type));
    const clang::TagType *tag_type = llvm::dyn_cast<clang::TagType>(qual_type);
    if (tag_type)
        return tag_type->isBeingDefined();
    return false;
}

bool
ClangASTContext::IsObjCObjectPointerType (const CompilerType& type, CompilerType *class_type_ptr)
{
    if (!type)
        return false;

    clang::QualType qual_type (GetCanonicalQualType(type));

    if (!qual_type.isNull() && qual_type->isObjCObjectPointerType())
    {
        if (class_type_ptr)
        {
            if (!qual_type->isObjCClassType() &&
                !qual_type->isObjCIdType())
            {
                const clang::ObjCObjectPointerType *obj_pointer_type = llvm::dyn_cast<clang::ObjCObjectPointerType>(qual_type);
                if (obj_pointer_type == nullptr)
                    class_type_ptr->Clear();
                else
                    class_type_ptr->SetCompilerType (type.GetTypeSystem(), clang::QualType(obj_pointer_type->getInterfaceType(), 0).getAsOpaquePtr());
            }
        }
        return true;
    }
    if (class_type_ptr)
        class_type_ptr->Clear();
    return false;
}

bool
ClangASTContext::GetObjCClassName (const CompilerType& type, std::string &class_name)
{
    if (!type)
        return false;
    
    clang::QualType qual_type (GetCanonicalQualType(type));
    
    const clang::ObjCObjectType *object_type = llvm::dyn_cast<clang::ObjCObjectType>(qual_type);
    if (object_type)
    {
        const clang::ObjCInterfaceDecl *interface = object_type->getInterface();
        if (interface)
        {
            class_name = interface->getNameAsString();
            return true;
        }
    }
    return false;
}


//----------------------------------------------------------------------
// Type Completion
//----------------------------------------------------------------------

bool
ClangASTContext::GetCompleteType (lldb::opaque_compiler_type_t type)
{
    if (!type)
        return false;
    const bool allow_completion = true;
    return GetCompleteQualType (getASTContext(), GetQualType(type), allow_completion);
}

ConstString
ClangASTContext::GetTypeName (lldb::opaque_compiler_type_t type)
{
    std::string type_name;
    if (type)
    {
        clang::PrintingPolicy printing_policy (getASTContext()->getPrintingPolicy());
        clang::QualType qual_type(GetQualType(type));
        printing_policy.SuppressTagKeyword = true;
        printing_policy.LangOpts.WChar = true;
        const clang::TypedefType *typedef_type = qual_type->getAs<clang::TypedefType>();
        if (typedef_type)
        {
            const clang::TypedefNameDecl *typedef_decl = typedef_type->getDecl();
            type_name = typedef_decl->getQualifiedNameAsString();
        }
        else
        {
            type_name = qual_type.getAsString(printing_policy);
        }
    }
    return ConstString(type_name);
}

uint32_t
ClangASTContext::GetTypeInfo (lldb::opaque_compiler_type_t type, CompilerType *pointee_or_element_clang_type)
{
    if (!type)
        return 0;
    
    if (pointee_or_element_clang_type)
        pointee_or_element_clang_type->Clear();
    
    clang::QualType qual_type (GetQualType(type));
    
    const clang::Type::TypeClass type_class = qual_type->getTypeClass();
    switch (type_class)
    {
        case clang::Type::Builtin:
        {
            const clang::BuiltinType *builtin_type = llvm::dyn_cast<clang::BuiltinType>(qual_type->getCanonicalTypeInternal());
            
            uint32_t builtin_type_flags = eTypeIsBuiltIn | eTypeHasValue;
            switch (builtin_type->getKind())
            {
                case clang::BuiltinType::ObjCId:
                case clang::BuiltinType::ObjCClass:
                    if (pointee_or_element_clang_type)
                        pointee_or_element_clang_type->SetCompilerType(getASTContext(), getASTContext()->ObjCBuiltinClassTy);
                    builtin_type_flags |= eTypeIsPointer | eTypeIsObjC;
                    break;
                    
                case clang::BuiltinType::ObjCSel:
                    if (pointee_or_element_clang_type)
                        pointee_or_element_clang_type->SetCompilerType(getASTContext(), getASTContext()->CharTy);
                    builtin_type_flags |= eTypeIsPointer | eTypeIsObjC;
                    break;
                    
                case clang::BuiltinType::Bool:
                case clang::BuiltinType::Char_U:
                case clang::BuiltinType::UChar:
                case clang::BuiltinType::WChar_U:
                case clang::BuiltinType::Char16:
                case clang::BuiltinType::Char32:
                case clang::BuiltinType::UShort:
                case clang::BuiltinType::UInt:
                case clang::BuiltinType::ULong:
                case clang::BuiltinType::ULongLong:
                case clang::BuiltinType::UInt128:
                case clang::BuiltinType::Char_S:
                case clang::BuiltinType::SChar:
                case clang::BuiltinType::WChar_S:
                case clang::BuiltinType::Short:
                case clang::BuiltinType::Int:
                case clang::BuiltinType::Long:
                case clang::BuiltinType::LongLong:
                case clang::BuiltinType::Int128:
                case clang::BuiltinType::Float:
                case clang::BuiltinType::Double:
                case clang::BuiltinType::LongDouble:
                    builtin_type_flags |= eTypeIsScalar;
                    if (builtin_type->isInteger())
                    {
                        builtin_type_flags |= eTypeIsInteger;
                        if (builtin_type->isSignedInteger())
                            builtin_type_flags |= eTypeIsSigned;
                    }
                    else if (builtin_type->isFloatingPoint())
                        builtin_type_flags |= eTypeIsFloat;
                    break;
                default:
                    break;
            }
            return builtin_type_flags;
        }
            
        case clang::Type::BlockPointer:
            if (pointee_or_element_clang_type)
                pointee_or_element_clang_type->SetCompilerType(getASTContext(), qual_type->getPointeeType());
            return eTypeIsPointer | eTypeHasChildren | eTypeIsBlock;
            
        case clang::Type::Complex:
        {
            uint32_t complex_type_flags = eTypeIsBuiltIn | eTypeHasValue | eTypeIsComplex;
            const clang::ComplexType *complex_type = llvm::dyn_cast<clang::ComplexType>(qual_type->getCanonicalTypeInternal());
            if (complex_type)
            {
                clang::QualType complex_element_type (complex_type->getElementType());
                if (complex_element_type->isIntegerType())
                    complex_type_flags |= eTypeIsFloat;
                else if (complex_element_type->isFloatingType())
                    complex_type_flags |= eTypeIsInteger;
            }
            return complex_type_flags;
        }
            break;
            
        case clang::Type::ConstantArray:
        case clang::Type::DependentSizedArray:
        case clang::Type::IncompleteArray:
        case clang::Type::VariableArray:
            if (pointee_or_element_clang_type)
                pointee_or_element_clang_type->SetCompilerType(getASTContext(), llvm::cast<clang::ArrayType>(qual_type.getTypePtr())->getElementType());
            return eTypeHasChildren | eTypeIsArray;
            
        case clang::Type::DependentName:                    return 0;
        case clang::Type::DependentSizedExtVector:          return eTypeHasChildren | eTypeIsVector;
        case clang::Type::DependentTemplateSpecialization:  return eTypeIsTemplate;
        case clang::Type::Decltype:                         return 0;
            
        case clang::Type::Enum:
            if (pointee_or_element_clang_type)
                pointee_or_element_clang_type->SetCompilerType(getASTContext(), llvm::cast<clang::EnumType>(qual_type)->getDecl()->getIntegerType());
            return eTypeIsEnumeration | eTypeHasValue;

        case clang::Type::Auto:
            return CompilerType (getASTContext(), llvm::cast<clang::AutoType>(qual_type)->getDeducedType()).GetTypeInfo (pointee_or_element_clang_type);
        case clang::Type::Elaborated:
            return CompilerType (getASTContext(), llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType()).GetTypeInfo (pointee_or_element_clang_type);
        case clang::Type::Paren:
            return CompilerType (getASTContext(), llvm::cast<clang::ParenType>(qual_type)->desugar()).GetTypeInfo (pointee_or_element_clang_type);
            
        case clang::Type::FunctionProto:                    return eTypeIsFuncPrototype | eTypeHasValue;
        case clang::Type::FunctionNoProto:                  return eTypeIsFuncPrototype | eTypeHasValue;
        case clang::Type::InjectedClassName:                return 0;
            
        case clang::Type::LValueReference:
        case clang::Type::RValueReference:
            if (pointee_or_element_clang_type)
                pointee_or_element_clang_type->SetCompilerType(getASTContext(), llvm::cast<clang::ReferenceType>(qual_type.getTypePtr())->getPointeeType());
            return eTypeHasChildren | eTypeIsReference | eTypeHasValue;
            
        case clang::Type::MemberPointer:                    return eTypeIsPointer   | eTypeIsMember | eTypeHasValue;
            
        case clang::Type::ObjCObjectPointer:
            if (pointee_or_element_clang_type)
                pointee_or_element_clang_type->SetCompilerType(getASTContext(), qual_type->getPointeeType());
            return eTypeHasChildren | eTypeIsObjC | eTypeIsClass | eTypeIsPointer | eTypeHasValue;
            
        case clang::Type::ObjCObject:                       return eTypeHasChildren | eTypeIsObjC | eTypeIsClass;
        case clang::Type::ObjCInterface:                    return eTypeHasChildren | eTypeIsObjC | eTypeIsClass;
            
        case clang::Type::Pointer:
            if (pointee_or_element_clang_type)
                pointee_or_element_clang_type->SetCompilerType(getASTContext(), qual_type->getPointeeType());
            return eTypeHasChildren | eTypeIsPointer | eTypeHasValue;
            
        case clang::Type::Record:
            if (qual_type->getAsCXXRecordDecl())
                return eTypeHasChildren | eTypeIsClass | eTypeIsCPlusPlus;
            else
                return eTypeHasChildren | eTypeIsStructUnion;
            break;
        case clang::Type::SubstTemplateTypeParm:            return eTypeIsTemplate;
        case clang::Type::TemplateTypeParm:                 return eTypeIsTemplate;
        case clang::Type::TemplateSpecialization:           return eTypeIsTemplate;
            
        case clang::Type::Typedef:
            return eTypeIsTypedef | CompilerType (getASTContext(), llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType()).GetTypeInfo (pointee_or_element_clang_type);
        case clang::Type::TypeOfExpr:                       return 0;
        case clang::Type::TypeOf:                           return 0;
        case clang::Type::UnresolvedUsing:                  return 0;
            
        case clang::Type::ExtVector:
        case clang::Type::Vector:
        {
            uint32_t vector_type_flags = eTypeHasChildren | eTypeIsVector;
            const clang::VectorType *vector_type = llvm::dyn_cast<clang::VectorType>(qual_type->getCanonicalTypeInternal());
            if (vector_type)
            {
                if (vector_type->isIntegerType())
                    vector_type_flags |= eTypeIsFloat;
                else if (vector_type->isFloatingType())
                    vector_type_flags |= eTypeIsInteger;
            }
            return vector_type_flags;
        }
        default:                                            return 0;
    }
    return 0;
}



lldb::LanguageType
ClangASTContext::GetMinimumLanguage (lldb::opaque_compiler_type_t type)
{
    if (!type)
        return lldb::eLanguageTypeC;
    
    // If the type is a reference, then resolve it to what it refers to first:
    clang::QualType qual_type (GetCanonicalQualType(type).getNonReferenceType());
    if (qual_type->isAnyPointerType())
    {
        if (qual_type->isObjCObjectPointerType())
            return lldb::eLanguageTypeObjC;
        
        clang::QualType pointee_type (qual_type->getPointeeType());
        if (pointee_type->getPointeeCXXRecordDecl() != nullptr)
            return lldb::eLanguageTypeC_plus_plus;
        if (pointee_type->isObjCObjectOrInterfaceType())
            return lldb::eLanguageTypeObjC;
        if (pointee_type->isObjCClassType())
            return lldb::eLanguageTypeObjC;
        if (pointee_type.getTypePtr() == getASTContext()->ObjCBuiltinIdTy.getTypePtr())
            return lldb::eLanguageTypeObjC;
    }
    else
    {
        if (qual_type->isObjCObjectOrInterfaceType())
            return lldb::eLanguageTypeObjC;
        if (qual_type->getAsCXXRecordDecl())
            return lldb::eLanguageTypeC_plus_plus;
        switch (qual_type->getTypeClass())
        {
            default:
                break;
            case clang::Type::Builtin:
                switch (llvm::cast<clang::BuiltinType>(qual_type)->getKind())
            {
                default:
                case clang::BuiltinType::Void:
                case clang::BuiltinType::Bool:
                case clang::BuiltinType::Char_U:
                case clang::BuiltinType::UChar:
                case clang::BuiltinType::WChar_U:
                case clang::BuiltinType::Char16:
                case clang::BuiltinType::Char32:
                case clang::BuiltinType::UShort:
                case clang::BuiltinType::UInt:
                case clang::BuiltinType::ULong:
                case clang::BuiltinType::ULongLong:
                case clang::BuiltinType::UInt128:
                case clang::BuiltinType::Char_S:
                case clang::BuiltinType::SChar:
                case clang::BuiltinType::WChar_S:
                case clang::BuiltinType::Short:
                case clang::BuiltinType::Int:
                case clang::BuiltinType::Long:
                case clang::BuiltinType::LongLong:
                case clang::BuiltinType::Int128:
                case clang::BuiltinType::Float:
                case clang::BuiltinType::Double:
                case clang::BuiltinType::LongDouble:
                    break;
                    
                case clang::BuiltinType::NullPtr:
                    return eLanguageTypeC_plus_plus;
                    
                case clang::BuiltinType::ObjCId:
                case clang::BuiltinType::ObjCClass:
                case clang::BuiltinType::ObjCSel:
                    return eLanguageTypeObjC;
                    
                case clang::BuiltinType::Dependent:
                case clang::BuiltinType::Overload:
                case clang::BuiltinType::BoundMember:
                case clang::BuiltinType::UnknownAny:
                    break;
            }
                break;
            case clang::Type::Typedef:
                return CompilerType(getASTContext(), llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType()).GetMinimumLanguage();
        }
    }
    return lldb::eLanguageTypeC;
}

lldb::TypeClass
ClangASTContext::GetTypeClass (lldb::opaque_compiler_type_t type)
{
    if (!type)
        return lldb::eTypeClassInvalid;
    
    clang::QualType qual_type(GetQualType(type));
    
    switch (qual_type->getTypeClass())
    {
        case clang::Type::UnaryTransform:           break;
        case clang::Type::FunctionNoProto:          return lldb::eTypeClassFunction;
        case clang::Type::FunctionProto:            return lldb::eTypeClassFunction;
        case clang::Type::IncompleteArray:          return lldb::eTypeClassArray;
        case clang::Type::VariableArray:            return lldb::eTypeClassArray;
        case clang::Type::ConstantArray:            return lldb::eTypeClassArray;
        case clang::Type::DependentSizedArray:      return lldb::eTypeClassArray;
        case clang::Type::DependentSizedExtVector:  return lldb::eTypeClassVector;
        case clang::Type::ExtVector:                return lldb::eTypeClassVector;
        case clang::Type::Vector:                   return lldb::eTypeClassVector;
        case clang::Type::Builtin:                  return lldb::eTypeClassBuiltin;
        case clang::Type::ObjCObjectPointer:        return lldb::eTypeClassObjCObjectPointer;
        case clang::Type::BlockPointer:             return lldb::eTypeClassBlockPointer;
        case clang::Type::Pointer:                  return lldb::eTypeClassPointer;
        case clang::Type::LValueReference:          return lldb::eTypeClassReference;
        case clang::Type::RValueReference:          return lldb::eTypeClassReference;
        case clang::Type::MemberPointer:            return lldb::eTypeClassMemberPointer;
        case clang::Type::Complex:
            if (qual_type->isComplexType())
                return lldb::eTypeClassComplexFloat;
            else
                return lldb::eTypeClassComplexInteger;
        case clang::Type::ObjCObject:               return lldb::eTypeClassObjCObject;
        case clang::Type::ObjCInterface:            return lldb::eTypeClassObjCInterface;
        case clang::Type::Record:
        {
            const clang::RecordType *record_type = llvm::cast<clang::RecordType>(qual_type.getTypePtr());
            const clang::RecordDecl *record_decl = record_type->getDecl();
            if (record_decl->isUnion())
                return lldb::eTypeClassUnion;
            else if (record_decl->isStruct())
                return lldb::eTypeClassStruct;
            else
                return lldb::eTypeClassClass;
        }
            break;
        case clang::Type::Enum:                     return lldb::eTypeClassEnumeration;
        case clang::Type::Typedef:                  return lldb::eTypeClassTypedef;
        case clang::Type::UnresolvedUsing:          break;
        case clang::Type::Paren:
            return CompilerType(getASTContext(), llvm::cast<clang::ParenType>(qual_type)->desugar()).GetTypeClass();
        case clang::Type::Auto:
            return CompilerType(getASTContext(), llvm::cast<clang::AutoType>(qual_type)->getDeducedType()).GetTypeClass();
        case clang::Type::Elaborated:
            return CompilerType(getASTContext(), llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType()).GetTypeClass();
            
        case clang::Type::Attributed:               break;
        case clang::Type::TemplateTypeParm:         break;
        case clang::Type::SubstTemplateTypeParm:    break;
        case clang::Type::SubstTemplateTypeParmPack:break;
        case clang::Type::InjectedClassName:        break;
        case clang::Type::DependentName:            break;
        case clang::Type::DependentTemplateSpecialization: break;
        case clang::Type::PackExpansion:            break;
            
        case clang::Type::TypeOfExpr:               break;
        case clang::Type::TypeOf:                   break;
        case clang::Type::Decltype:                 break;
        case clang::Type::TemplateSpecialization:   break;
        case clang::Type::Atomic:                   break;
            
            // pointer type decayed from an array or function type.
        case clang::Type::Decayed:                  break;
        case clang::Type::Adjusted:                 break;
    }
    // We don't know hot to display this type...
    return lldb::eTypeClassOther;
    
}

unsigned
ClangASTContext::GetTypeQualifiers(lldb::opaque_compiler_type_t type)
{
    if (type)
        return GetQualType(type).getQualifiers().getCVRQualifiers();
    return 0;
}

//----------------------------------------------------------------------
// Creating related types
//----------------------------------------------------------------------

CompilerType
ClangASTContext::GetArrayElementType (lldb::opaque_compiler_type_t type, uint64_t *stride)
{
    if (type)
    {
        clang::QualType qual_type(GetCanonicalQualType(type));
        
        const clang::Type *array_eletype = qual_type.getTypePtr()->getArrayElementTypeNoTypeQual();
        
        if (!array_eletype)
            return CompilerType();
        
        CompilerType element_type (getASTContext(), array_eletype->getCanonicalTypeUnqualified());
        
        // TODO: the real stride will be >= this value.. find the real one!
        if (stride)
            *stride = element_type.GetByteSize(nullptr);
        
        return element_type;
        
    }
    return CompilerType();
}

CompilerType
ClangASTContext::GetCanonicalType (lldb::opaque_compiler_type_t type)
{
    if (type)
        return CompilerType (getASTContext(), GetCanonicalQualType(type));
    return CompilerType();
}

static clang::QualType
GetFullyUnqualifiedType_Impl (clang::ASTContext *ast, clang::QualType qual_type)
{
    if (qual_type->isPointerType())
        qual_type = ast->getPointerType(GetFullyUnqualifiedType_Impl(ast, qual_type->getPointeeType()));
    else
        qual_type = qual_type.getUnqualifiedType();
    qual_type.removeLocalConst();
    qual_type.removeLocalRestrict();
    qual_type.removeLocalVolatile();
    return qual_type;
}

CompilerType
ClangASTContext::GetFullyUnqualifiedType (lldb::opaque_compiler_type_t type)
{
    if (type)
        return CompilerType(getASTContext(), GetFullyUnqualifiedType_Impl(getASTContext(), GetQualType(type)));
    return CompilerType();
}


int
ClangASTContext::GetFunctionArgumentCount (lldb::opaque_compiler_type_t type)
{
    if (type)
    {
        const clang::FunctionProtoType* func = llvm::dyn_cast<clang::FunctionProtoType>(GetCanonicalQualType(type));
        if (func)
            return func->getNumParams();
    }
    return -1;
}

CompilerType
ClangASTContext::GetFunctionArgumentTypeAtIndex (lldb::opaque_compiler_type_t type, size_t idx)
{
    if (type)
    {
        const clang::FunctionProtoType* func = llvm::dyn_cast<clang::FunctionProtoType>(GetQualType(type));
        if (func)
        {
            const uint32_t num_args = func->getNumParams();
            if (idx < num_args)
                return CompilerType(getASTContext(), func->getParamType(idx));
        }
    }
    return CompilerType();
}

CompilerType
ClangASTContext::GetFunctionReturnType (lldb::opaque_compiler_type_t type)
{
    if (type)
    {
        clang::QualType qual_type(GetQualType(type));
        const clang::FunctionProtoType* func = llvm::dyn_cast<clang::FunctionProtoType>(qual_type.getTypePtr());
        if (func)
            return CompilerType(getASTContext(), func->getReturnType());
    }
    return CompilerType();
}

size_t
ClangASTContext::GetNumMemberFunctions (lldb::opaque_compiler_type_t type)
{
    size_t num_functions = 0;
    if (type)
    {
        clang::QualType qual_type(GetCanonicalQualType(type));
        switch (qual_type->getTypeClass()) {
            case clang::Type::Record:
                if (GetCompleteQualType (getASTContext(), qual_type))
                {
                    const clang::RecordType *record_type = llvm::cast<clang::RecordType>(qual_type.getTypePtr());
                    const clang::RecordDecl *record_decl = record_type->getDecl();
                    assert(record_decl);
                    const clang::CXXRecordDecl *cxx_record_decl = llvm::dyn_cast<clang::CXXRecordDecl>(record_decl);
                    if (cxx_record_decl)
                        num_functions = std::distance(cxx_record_decl->method_begin(), cxx_record_decl->method_end());
                }
                break;
                
            case clang::Type::ObjCObjectPointer:
                if (GetCompleteType(type))
                {
                    const clang::ObjCObjectPointerType *objc_class_type = qual_type->getAsObjCInterfacePointerType();
                    if (objc_class_type)
                    {
                        clang::ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterfaceDecl();
                        if (class_interface_decl)
                            num_functions = std::distance(class_interface_decl->meth_begin(), class_interface_decl->meth_end());
                    }
                }
                break;
                
            case clang::Type::ObjCObject:
            case clang::Type::ObjCInterface:
                if (GetCompleteType(type))
                {
                    const clang::ObjCObjectType *objc_class_type = llvm::dyn_cast<clang::ObjCObjectType>(qual_type.getTypePtr());
                    if (objc_class_type)
                    {
                        clang::ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterface();
                        if (class_interface_decl)
                            num_functions = std::distance(class_interface_decl->meth_begin(), class_interface_decl->meth_end());
                    }
                }
                break;
                
                
            case clang::Type::Typedef:
                return CompilerType (getASTContext(), llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType()).GetNumMemberFunctions();

            case clang::Type::Auto:
                return CompilerType (getASTContext(), llvm::cast<clang::AutoType>(qual_type)->getDeducedType()).GetNumMemberFunctions();
                
            case clang::Type::Elaborated:
                return CompilerType (getASTContext(), llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType()).GetNumMemberFunctions();
                
            case clang::Type::Paren:
                return CompilerType (getASTContext(), llvm::cast<clang::ParenType>(qual_type)->desugar()).GetNumMemberFunctions();
                
            default:
                break;
        }
    }
    return num_functions;
}

TypeMemberFunctionImpl
ClangASTContext::GetMemberFunctionAtIndex (lldb::opaque_compiler_type_t type, size_t idx)
{
    std::string name;
    MemberFunctionKind kind(MemberFunctionKind::eMemberFunctionKindUnknown);
    CompilerType clang_type;
    CompilerDecl clang_decl;
    if (type)
    {
        clang::QualType qual_type(GetCanonicalQualType(type));
        switch (qual_type->getTypeClass()) {
            case clang::Type::Record:
                if (GetCompleteQualType (getASTContext(), qual_type))
                {
                    const clang::RecordType *record_type = llvm::cast<clang::RecordType>(qual_type.getTypePtr());
                    const clang::RecordDecl *record_decl = record_type->getDecl();
                    assert(record_decl);
                    const clang::CXXRecordDecl *cxx_record_decl = llvm::dyn_cast<clang::CXXRecordDecl>(record_decl);
                    if (cxx_record_decl)
                    {
                        auto method_iter = cxx_record_decl->method_begin();
                        auto method_end = cxx_record_decl->method_end();
                        if (idx < static_cast<size_t>(std::distance(method_iter, method_end)))
                        {
                            std::advance(method_iter, idx);
                            clang::CXXMethodDecl *cxx_method_decl = method_iter->getCanonicalDecl();
                            if (cxx_method_decl)
                            {
                                name = cxx_method_decl->getDeclName().getAsString();
                                if (cxx_method_decl->isStatic())
                                    kind = lldb::eMemberFunctionKindStaticMethod;
                                else if (llvm::isa<clang::CXXConstructorDecl>(cxx_method_decl))
                                    kind = lldb::eMemberFunctionKindConstructor;
                                else if (llvm::isa<clang::CXXDestructorDecl>(cxx_method_decl))
                                    kind = lldb::eMemberFunctionKindDestructor;
                                else
                                    kind = lldb::eMemberFunctionKindInstanceMethod;
                                clang_type = CompilerType(this, cxx_method_decl->getType().getAsOpaquePtr());
                                clang_decl = CompilerDecl(this, cxx_method_decl);
                            }
                        }
                    }
                }
                break;
                
            case clang::Type::ObjCObjectPointer:
                if (GetCompleteType(type))
                {
                    const clang::ObjCObjectPointerType *objc_class_type = qual_type->getAsObjCInterfacePointerType();
                    if (objc_class_type)
                    {
                        clang::ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterfaceDecl();
                        if (class_interface_decl)
                        {
                            auto method_iter = class_interface_decl->meth_begin();
                            auto method_end = class_interface_decl->meth_end();
                            if (idx < static_cast<size_t>(std::distance(method_iter, method_end)))
                            {
                                std::advance(method_iter, idx);
                                clang::ObjCMethodDecl *objc_method_decl = method_iter->getCanonicalDecl();
                                if (objc_method_decl)
                                {
                                    clang_decl = CompilerDecl(this, objc_method_decl);
                                    name = objc_method_decl->getSelector().getAsString();
                                    if (objc_method_decl->isClassMethod())
                                        kind = lldb::eMemberFunctionKindStaticMethod;
                                    else
                                        kind = lldb::eMemberFunctionKindInstanceMethod;
                                }
                            }
                        }
                    }
                }
                break;
                
            case clang::Type::ObjCObject:
            case clang::Type::ObjCInterface:
                if (GetCompleteType(type))
                {
                    const clang::ObjCObjectType *objc_class_type = llvm::dyn_cast<clang::ObjCObjectType>(qual_type.getTypePtr());
                    if (objc_class_type)
                    {
                        clang::ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterface();
                        if (class_interface_decl)
                        {
                            auto method_iter = class_interface_decl->meth_begin();
                            auto method_end = class_interface_decl->meth_end();
                            if (idx < static_cast<size_t>(std::distance(method_iter, method_end)))
                            {
                                std::advance(method_iter, idx);
                                clang::ObjCMethodDecl *objc_method_decl = method_iter->getCanonicalDecl();
                                if (objc_method_decl)
                                {
                                    clang_decl = CompilerDecl(this, objc_method_decl);
                                    name = objc_method_decl->getSelector().getAsString();
                                    if (objc_method_decl->isClassMethod())
                                        kind = lldb::eMemberFunctionKindStaticMethod;
                                    else
                                        kind = lldb::eMemberFunctionKindInstanceMethod;
                                }
                            }
                        }
                    }
                }
                break;
                
            case clang::Type::Typedef:
                return GetMemberFunctionAtIndex(llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType().getAsOpaquePtr(), idx);

            case clang::Type::Auto:
                return GetMemberFunctionAtIndex(llvm::cast<clang::AutoType>(qual_type)->getDeducedType().getAsOpaquePtr(), idx);
                
            case clang::Type::Elaborated:
                return GetMemberFunctionAtIndex(llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType().getAsOpaquePtr(), idx);
                
            case clang::Type::Paren:
                return GetMemberFunctionAtIndex(llvm::cast<clang::ParenType>(qual_type)->desugar().getAsOpaquePtr(), idx);
                
            default:
                break;
        }
    }
    
    if (kind == eMemberFunctionKindUnknown)
        return TypeMemberFunctionImpl();
    else
        return TypeMemberFunctionImpl(clang_type, clang_decl, name, kind);
}

CompilerType
ClangASTContext::GetNonReferenceType (lldb::opaque_compiler_type_t type)
{
    if (type)
        return CompilerType(getASTContext(), GetQualType(type).getNonReferenceType());
    return CompilerType();
}

CompilerType
ClangASTContext::CreateTypedefType (const CompilerType& type,
                                    const char *typedef_name,
                                    const CompilerDeclContext &compiler_decl_ctx)
{
    if (type && typedef_name && typedef_name[0])
    {
        ClangASTContext *ast = llvm::dyn_cast<ClangASTContext>(type.GetTypeSystem());
        if (!ast)
            return CompilerType();
        clang::ASTContext* clang_ast = ast->getASTContext();
        clang::QualType qual_type (GetQualType(type));

        clang::DeclContext *decl_ctx = ClangASTContext::DeclContextGetAsDeclContext(compiler_decl_ctx);
        if (decl_ctx == nullptr)
            decl_ctx = ast->getASTContext()->getTranslationUnitDecl();

        clang::TypedefDecl *decl = clang::TypedefDecl::Create (*clang_ast,
                                                               decl_ctx,
                                                               clang::SourceLocation(),
                                                               clang::SourceLocation(),
                                                               &clang_ast->Idents.get(typedef_name),
                                                               clang_ast->getTrivialTypeSourceInfo(qual_type));
        
        decl->setAccess(clang::AS_public); // TODO respect proper access specifier
        
        // Get a uniqued clang::QualType for the typedef decl type
        return CompilerType (clang_ast, clang_ast->getTypedefType (decl));
    }
    return CompilerType();
    
}

CompilerType
ClangASTContext::GetPointeeType (lldb::opaque_compiler_type_t type)
{
    if (type)
    {
        clang::QualType qual_type(GetQualType(type));
        return CompilerType (getASTContext(), qual_type.getTypePtr()->getPointeeType());
    }
    return CompilerType();
}

CompilerType
ClangASTContext::GetPointerType (lldb::opaque_compiler_type_t type)
{
    if (type)
    {
        clang::QualType qual_type (GetQualType(type));
        
        const clang::Type::TypeClass type_class = qual_type->getTypeClass();
        switch (type_class)
        {
            case clang::Type::ObjCObject:
            case clang::Type::ObjCInterface:
                return CompilerType(getASTContext(), getASTContext()->getObjCObjectPointerType(qual_type));
                
            default:
                return CompilerType(getASTContext(), getASTContext()->getPointerType(qual_type));
        }
    }
    return CompilerType();
}


CompilerType
ClangASTContext::GetLValueReferenceType (lldb::opaque_compiler_type_t type)
{
    if (type)
        return CompilerType(this, getASTContext()->getLValueReferenceType(GetQualType(type)).getAsOpaquePtr());
    else
        return CompilerType();
}

CompilerType
ClangASTContext::GetRValueReferenceType (lldb::opaque_compiler_type_t type)
{
    if (type)
        return CompilerType(this, getASTContext()->getRValueReferenceType(GetQualType(type)).getAsOpaquePtr());
    else
        return CompilerType();
}

CompilerType
ClangASTContext::AddConstModifier (lldb::opaque_compiler_type_t type)
{
    if (type)
    {
        clang::QualType result(GetQualType(type));
        result.addConst();
        return CompilerType (this, result.getAsOpaquePtr());
    }
    return CompilerType();
}

CompilerType
ClangASTContext::AddVolatileModifier (lldb::opaque_compiler_type_t type)
{
    if (type)
    {
        clang::QualType result(GetQualType(type));
        result.addVolatile();
        return CompilerType (this, result.getAsOpaquePtr());
    }
    return CompilerType();

}

CompilerType
ClangASTContext::AddRestrictModifier (lldb::opaque_compiler_type_t type)
{
    if (type)
    {
        clang::QualType result(GetQualType(type));
        result.addRestrict();
        return CompilerType (this, result.getAsOpaquePtr());
    }
    return CompilerType();

}

CompilerType
ClangASTContext::CreateTypedef (lldb::opaque_compiler_type_t type, const char *typedef_name, const CompilerDeclContext &compiler_decl_ctx)
{
    if (type)
    {
        clang::ASTContext* clang_ast = getASTContext();
        clang::QualType qual_type (GetQualType(type));

        clang::DeclContext *decl_ctx = ClangASTContext::DeclContextGetAsDeclContext(compiler_decl_ctx);
        if (decl_ctx == nullptr)
            decl_ctx = getASTContext()->getTranslationUnitDecl();

        clang::TypedefDecl *decl = clang::TypedefDecl::Create (*clang_ast,
                                                               decl_ctx,
                                                               clang::SourceLocation(),
                                                               clang::SourceLocation(),
                                                               &clang_ast->Idents.get(typedef_name),
                                                               clang_ast->getTrivialTypeSourceInfo(qual_type));

        decl->setAccess(clang::AS_public); // TODO respect proper access specifier

        // Get a uniqued clang::QualType for the typedef decl type
        return CompilerType (this, clang_ast->getTypedefType (decl).getAsOpaquePtr());

    }
    return CompilerType();

}

CompilerType
ClangASTContext::GetTypedefedType (lldb::opaque_compiler_type_t type)
{
    if (type)
    {
        const clang::TypedefType *typedef_type = llvm::dyn_cast<clang::TypedefType>(GetQualType(type));
        if (typedef_type)
            return CompilerType (getASTContext(), typedef_type->getDecl()->getUnderlyingType());
    }
    return CompilerType();
}

CompilerType
ClangASTContext::RemoveFastQualifiers (const CompilerType& type)
{
    if (IsClangType(type))
    {
        clang::QualType qual_type(GetQualType(type));
        qual_type.getQualifiers().removeFastQualifiers();
        return CompilerType (type.GetTypeSystem(), qual_type.getAsOpaquePtr());
    }
    return type;
}


//----------------------------------------------------------------------
// Create related types using the current type's AST
//----------------------------------------------------------------------

CompilerType
ClangASTContext::GetBasicTypeFromAST (lldb::BasicType basic_type)
{
    return ClangASTContext::GetBasicType(getASTContext(), basic_type);
}
//----------------------------------------------------------------------
// Exploring the type
//----------------------------------------------------------------------

uint64_t
ClangASTContext::GetBitSize (lldb::opaque_compiler_type_t type, ExecutionContextScope *exe_scope)
{
    if (GetCompleteType (type))
    {
        clang::QualType qual_type(GetCanonicalQualType(type));
        switch (qual_type->getTypeClass())
        {
            case clang::Type::ObjCInterface:
            case clang::Type::ObjCObject:
            {
                ExecutionContext exe_ctx (exe_scope);
                Process *process = exe_ctx.GetProcessPtr();
                if (process)
                {
                    ObjCLanguageRuntime *objc_runtime = process->GetObjCLanguageRuntime();
                    if (objc_runtime)
                    {
                        uint64_t bit_size = 0;
                        if (objc_runtime->GetTypeBitSize(CompilerType(getASTContext(), qual_type), bit_size))
                            return bit_size;
                    }
                }
                else
                {
                    static bool g_printed = false;
                    if (!g_printed)
                    {
                        StreamString s;
                        DumpTypeDescription(type, &s);
                        
                        llvm::outs() << "warning: trying to determine the size of type ";
                        llvm::outs() << s.GetString() << "\n";
                        llvm::outs() << "without a valid ExecutionContext. this is not reliable. please file a bug against LLDB.\n";
                        llvm::outs() << "backtrace:\n";
                        llvm::sys::PrintStackTrace(llvm::outs());
                        llvm::outs() << "\n";
                        g_printed = true;
                    }
                }
            }
                // fallthrough
            default:
                const uint32_t bit_size = getASTContext()->getTypeSize (qual_type);
                if (bit_size == 0)
                {
                    if (qual_type->isIncompleteArrayType())
                        return getASTContext()->getTypeSize (qual_type->getArrayElementTypeNoTypeQual()->getCanonicalTypeUnqualified());
                }
                if (qual_type->isObjCObjectOrInterfaceType())
                    return bit_size + getASTContext()->getTypeSize(getASTContext()->ObjCBuiltinClassTy);
                return bit_size;
        }
    }
    return 0;
}

size_t
ClangASTContext::GetTypeBitAlign (lldb::opaque_compiler_type_t type)
{
    if (GetCompleteType(type))
        return getASTContext()->getTypeAlign(GetQualType(type));
    return 0;
}


lldb::Encoding
ClangASTContext::GetEncoding (lldb::opaque_compiler_type_t type, uint64_t &count)
{
    if (!type)
        return lldb::eEncodingInvalid;
    
    count = 1;
    clang::QualType qual_type(GetCanonicalQualType(type));
    
    switch (qual_type->getTypeClass())
    {
        case clang::Type::UnaryTransform:
            break;
            
        case clang::Type::FunctionNoProto:
        case clang::Type::FunctionProto:
            break;
            
        case clang::Type::IncompleteArray:
        case clang::Type::VariableArray:
            break;
            
        case clang::Type::ConstantArray:
            break;
            
        case clang::Type::ExtVector:
        case clang::Type::Vector:
            // TODO: Set this to more than one???
            break;
            
        case clang::Type::Builtin:
            switch (llvm::cast<clang::BuiltinType>(qual_type)->getKind())
        {
            case clang::BuiltinType::Void:
                break;
                
            case clang::BuiltinType::Bool:
            case clang::BuiltinType::Char_S:
            case clang::BuiltinType::SChar:
            case clang::BuiltinType::WChar_S:
            case clang::BuiltinType::Char16:
            case clang::BuiltinType::Char32:
            case clang::BuiltinType::Short:
            case clang::BuiltinType::Int:
            case clang::BuiltinType::Long:
            case clang::BuiltinType::LongLong:
            case clang::BuiltinType::Int128:        return lldb::eEncodingSint;
                
            case clang::BuiltinType::Char_U:
            case clang::BuiltinType::UChar:
            case clang::BuiltinType::WChar_U:
            case clang::BuiltinType::UShort:
            case clang::BuiltinType::UInt:
            case clang::BuiltinType::ULong:
            case clang::BuiltinType::ULongLong:
            case clang::BuiltinType::UInt128:       return lldb::eEncodingUint;
                
            case clang::BuiltinType::Half:
            case clang::BuiltinType::Float:
            case clang::BuiltinType::Double:
            case clang::BuiltinType::LongDouble:    return lldb::eEncodingIEEE754;
                
            case clang::BuiltinType::ObjCClass:
            case clang::BuiltinType::ObjCId:
            case clang::BuiltinType::ObjCSel:       return lldb::eEncodingUint;
                
            case clang::BuiltinType::NullPtr:       return lldb::eEncodingUint;
                
            case clang::BuiltinType::Kind::ARCUnbridgedCast:
            case clang::BuiltinType::Kind::BoundMember:
            case clang::BuiltinType::Kind::BuiltinFn:
            case clang::BuiltinType::Kind::Dependent:
            case clang::BuiltinType::Kind::OCLClkEvent:
            case clang::BuiltinType::Kind::OCLEvent:
            case clang::BuiltinType::Kind::OCLImage1d:
            case clang::BuiltinType::Kind::OCLImage1dArray:
            case clang::BuiltinType::Kind::OCLImage1dBuffer:
            case clang::BuiltinType::Kind::OCLImage2d:
            case clang::BuiltinType::Kind::OCLImage2dArray:
            case clang::BuiltinType::Kind::OCLImage2dArrayDepth:
            case clang::BuiltinType::Kind::OCLImage2dArrayMSAA:
            case clang::BuiltinType::Kind::OCLImage2dArrayMSAADepth:
            case clang::BuiltinType::Kind::OCLImage2dDepth:
            case clang::BuiltinType::Kind::OCLImage2dMSAA:
            case clang::BuiltinType::Kind::OCLImage2dMSAADepth:
            case clang::BuiltinType::Kind::OCLImage3d:
            case clang::BuiltinType::Kind::OCLQueue:
            case clang::BuiltinType::Kind::OCLNDRange:
            case clang::BuiltinType::Kind::OCLReserveID:
            case clang::BuiltinType::Kind::OCLSampler:
            case clang::BuiltinType::Kind::OMPArraySection:
            case clang::BuiltinType::Kind::Overload:
            case clang::BuiltinType::Kind::PseudoObject:
            case clang::BuiltinType::Kind::UnknownAny:
                break;
        }
            break;
            // All pointer types are represented as unsigned integer encodings.
            // We may nee to add a eEncodingPointer if we ever need to know the
            // difference
        case clang::Type::ObjCObjectPointer:
        case clang::Type::BlockPointer:
        case clang::Type::Pointer:
        case clang::Type::LValueReference:
        case clang::Type::RValueReference:
        case clang::Type::MemberPointer:            return lldb::eEncodingUint;
        case clang::Type::Complex:
        {
            lldb::Encoding encoding = lldb::eEncodingIEEE754;
            if (qual_type->isComplexType())
                encoding = lldb::eEncodingIEEE754;
            else
            {
                const clang::ComplexType *complex_type = qual_type->getAsComplexIntegerType();
                if (complex_type)
                    encoding = CompilerType(getASTContext(), complex_type->getElementType()).GetEncoding(count);
                else
                    encoding = lldb::eEncodingSint;
            }
            count = 2;
            return encoding;
        }
            
        case clang::Type::ObjCInterface:            break;
        case clang::Type::Record:                   break;
        case clang::Type::Enum:                     return lldb::eEncodingSint;
        case clang::Type::Typedef:
            return CompilerType(getASTContext(), llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType()).GetEncoding(count);

        case clang::Type::Auto:
            return CompilerType(getASTContext(), llvm::cast<clang::AutoType>(qual_type)->getDeducedType()).GetEncoding(count);
            
        case clang::Type::Elaborated:
            return CompilerType(getASTContext(), llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType()).GetEncoding(count);
            
        case clang::Type::Paren:
            return CompilerType(getASTContext(), llvm::cast<clang::ParenType>(qual_type)->desugar()).GetEncoding(count);
            
        case clang::Type::DependentSizedArray:
        case clang::Type::DependentSizedExtVector:
        case clang::Type::UnresolvedUsing:
        case clang::Type::Attributed:
        case clang::Type::TemplateTypeParm:
        case clang::Type::SubstTemplateTypeParm:
        case clang::Type::SubstTemplateTypeParmPack:
        case clang::Type::InjectedClassName:
        case clang::Type::DependentName:
        case clang::Type::DependentTemplateSpecialization:
        case clang::Type::PackExpansion:
        case clang::Type::ObjCObject:
            
        case clang::Type::TypeOfExpr:
        case clang::Type::TypeOf:
        case clang::Type::Decltype:
        case clang::Type::TemplateSpecialization:
        case clang::Type::Atomic:
        case clang::Type::Adjusted:
            break;
            
            // pointer type decayed from an array or function type.
        case clang::Type::Decayed:
            break;
    }
    count = 0;
    return lldb::eEncodingInvalid;
}

lldb::Format
ClangASTContext::GetFormat (lldb::opaque_compiler_type_t type)
{
    if (!type)
        return lldb::eFormatDefault;
    
    clang::QualType qual_type(GetCanonicalQualType(type));
    
    switch (qual_type->getTypeClass())
    {
        case clang::Type::UnaryTransform:
            break;
            
        case clang::Type::FunctionNoProto:
        case clang::Type::FunctionProto:
            break;
            
        case clang::Type::IncompleteArray:
        case clang::Type::VariableArray:
            break;
            
        case clang::Type::ConstantArray:
            return lldb::eFormatVoid; // no value
            
        case clang::Type::ExtVector:
        case clang::Type::Vector:
            break;
            
        case clang::Type::Builtin:
            switch (llvm::cast<clang::BuiltinType>(qual_type)->getKind())
        {
                //default: assert(0 && "Unknown builtin type!");
            case clang::BuiltinType::UnknownAny:
            case clang::BuiltinType::Void:
            case clang::BuiltinType::BoundMember:
                break;
                
            case clang::BuiltinType::Bool:          return lldb::eFormatBoolean;
            case clang::BuiltinType::Char_S:
            case clang::BuiltinType::SChar:
            case clang::BuiltinType::WChar_S:
            case clang::BuiltinType::Char_U:
            case clang::BuiltinType::UChar:
            case clang::BuiltinType::WChar_U:       return lldb::eFormatChar;
            case clang::BuiltinType::Char16:        return lldb::eFormatUnicode16;
            case clang::BuiltinType::Char32:        return lldb::eFormatUnicode32;
            case clang::BuiltinType::UShort:        return lldb::eFormatUnsigned;
            case clang::BuiltinType::Short:         return lldb::eFormatDecimal;
            case clang::BuiltinType::UInt:          return lldb::eFormatUnsigned;
            case clang::BuiltinType::Int:           return lldb::eFormatDecimal;
            case clang::BuiltinType::ULong:         return lldb::eFormatUnsigned;
            case clang::BuiltinType::Long:          return lldb::eFormatDecimal;
            case clang::BuiltinType::ULongLong:     return lldb::eFormatUnsigned;
            case clang::BuiltinType::LongLong:      return lldb::eFormatDecimal;
            case clang::BuiltinType::UInt128:       return lldb::eFormatUnsigned;
            case clang::BuiltinType::Int128:        return lldb::eFormatDecimal;
            case clang::BuiltinType::Half:
            case clang::BuiltinType::Float:
            case clang::BuiltinType::Double:
            case clang::BuiltinType::LongDouble:    return lldb::eFormatFloat;
            default:
                return lldb::eFormatHex;
        }
            break;
        case clang::Type::ObjCObjectPointer:        return lldb::eFormatHex;
        case clang::Type::BlockPointer:             return lldb::eFormatHex;
        case clang::Type::Pointer:                  return lldb::eFormatHex;
        case clang::Type::LValueReference:
        case clang::Type::RValueReference:          return lldb::eFormatHex;
        case clang::Type::MemberPointer:            break;
        case clang::Type::Complex:
        {
            if (qual_type->isComplexType())
                return lldb::eFormatComplex;
            else
                return lldb::eFormatComplexInteger;
        }
        case clang::Type::ObjCInterface:            break;
        case clang::Type::Record:                   break;
        case clang::Type::Enum:                     return lldb::eFormatEnum;
        case clang::Type::Typedef:
            return CompilerType (getASTContext(), llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType()).GetFormat();
        case clang::Type::Auto:
            return CompilerType (getASTContext(), llvm::cast<clang::AutoType>(qual_type)->desugar()).GetFormat();
        case clang::Type::Paren:
            return CompilerType (getASTContext(), llvm::cast<clang::ParenType>(qual_type)->desugar()).GetFormat();
        case clang::Type::Elaborated:
            return CompilerType (getASTContext(), llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType()).GetFormat();
        case clang::Type::DependentSizedArray:
        case clang::Type::DependentSizedExtVector:
        case clang::Type::UnresolvedUsing:
        case clang::Type::Attributed:
        case clang::Type::TemplateTypeParm:
        case clang::Type::SubstTemplateTypeParm:
        case clang::Type::SubstTemplateTypeParmPack:
        case clang::Type::InjectedClassName:
        case clang::Type::DependentName:
        case clang::Type::DependentTemplateSpecialization:
        case clang::Type::PackExpansion:
        case clang::Type::ObjCObject:
            
        case clang::Type::TypeOfExpr:
        case clang::Type::TypeOf:
        case clang::Type::Decltype:
        case clang::Type::TemplateSpecialization:
        case clang::Type::Atomic:
        case clang::Type::Adjusted:
            break;
            
            // pointer type decayed from an array or function type.
        case clang::Type::Decayed:
            break;
    }
    // We don't know hot to display this type...
    return lldb::eFormatBytes;
}

static bool
ObjCDeclHasIVars (clang::ObjCInterfaceDecl *class_interface_decl, bool check_superclass)
{
    while (class_interface_decl)
    {
        if (class_interface_decl->ivar_size() > 0)
            return true;
        
        if (check_superclass)
            class_interface_decl = class_interface_decl->getSuperClass();
        else
            break;
    }
    return false;
}

uint32_t
ClangASTContext::GetNumChildren (lldb::opaque_compiler_type_t type, bool omit_empty_base_classes)
{
    if (!type)
        return 0;
    
    uint32_t num_children = 0;
    clang::QualType qual_type(GetQualType(type));
    const clang::Type::TypeClass type_class = qual_type->getTypeClass();
    switch (type_class)
    {
        case clang::Type::Builtin:
            switch (llvm::cast<clang::BuiltinType>(qual_type)->getKind())
        {
            case clang::BuiltinType::ObjCId:    // child is Class
            case clang::BuiltinType::ObjCClass: // child is Class
                num_children = 1;
                break;
                
            default:
                break;
        }
            break;
            
        case clang::Type::Complex: return 0;
            
        case clang::Type::Record:
            if (GetCompleteQualType (getASTContext(), qual_type))
            {
                const clang::RecordType *record_type = llvm::cast<clang::RecordType>(qual_type.getTypePtr());
                const clang::RecordDecl *record_decl = record_type->getDecl();
                assert(record_decl);
                const clang::CXXRecordDecl *cxx_record_decl = llvm::dyn_cast<clang::CXXRecordDecl>(record_decl);
                if (cxx_record_decl)
                {
                    if (omit_empty_base_classes)
                    {
                        // Check each base classes to see if it or any of its
                        // base classes contain any fields. This can help
                        // limit the noise in variable views by not having to
                        // show base classes that contain no members.
                        clang::CXXRecordDecl::base_class_const_iterator base_class, base_class_end;
                        for (base_class = cxx_record_decl->bases_begin(), base_class_end = cxx_record_decl->bases_end();
                             base_class != base_class_end;
                             ++base_class)
                        {
                            const clang::CXXRecordDecl *base_class_decl = llvm::cast<clang::CXXRecordDecl>(base_class->getType()->getAs<clang::RecordType>()->getDecl());
                            
                            // Skip empty base classes
                            if (ClangASTContext::RecordHasFields(base_class_decl) == false)
                                continue;
                            
                            num_children++;
                        }
                    }
                    else
                    {
                        // Include all base classes
                        num_children += cxx_record_decl->getNumBases();
                    }
                    
                }
                clang::RecordDecl::field_iterator field, field_end;
                for (field = record_decl->field_begin(), field_end = record_decl->field_end(); field != field_end; ++field)
                    ++num_children;
            }
            break;
            
        case clang::Type::ObjCObject:
        case clang::Type::ObjCInterface:
            if (GetCompleteQualType (getASTContext(), qual_type))
            {
                const clang::ObjCObjectType *objc_class_type = llvm::dyn_cast<clang::ObjCObjectType>(qual_type.getTypePtr());
                assert (objc_class_type);
                if (objc_class_type)
                {
                    clang::ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterface();
                    
                    if (class_interface_decl)
                    {
                        
                        clang::ObjCInterfaceDecl *superclass_interface_decl = class_interface_decl->getSuperClass();
                        if (superclass_interface_decl)
                        {
                            if (omit_empty_base_classes)
                            {
                                if (ObjCDeclHasIVars (superclass_interface_decl, true))
                                    ++num_children;
                            }
                            else
                                ++num_children;
                        }
                        
                        num_children += class_interface_decl->ivar_size();
                    }
                }
            }
            break;
            
        case clang::Type::ObjCObjectPointer:
        {
            const clang::ObjCObjectPointerType *pointer_type = llvm::cast<clang::ObjCObjectPointerType>(qual_type.getTypePtr());
            clang::QualType pointee_type = pointer_type->getPointeeType();
            uint32_t num_pointee_children = CompilerType (getASTContext(),pointee_type).GetNumChildren (omit_empty_base_classes);
            // If this type points to a simple type, then it has 1 child
            if (num_pointee_children == 0)
                num_children = 1;
            else
                num_children = num_pointee_children;
        }
            break;
            
        case clang::Type::Vector:
        case clang::Type::ExtVector:
            num_children = llvm::cast<clang::VectorType>(qual_type.getTypePtr())->getNumElements();
            break;
            
        case clang::Type::ConstantArray:
            num_children = llvm::cast<clang::ConstantArrayType>(qual_type.getTypePtr())->getSize().getLimitedValue();
            break;
            
        case clang::Type::Pointer:
        {
            const clang::PointerType *pointer_type = llvm::cast<clang::PointerType>(qual_type.getTypePtr());
            clang::QualType pointee_type (pointer_type->getPointeeType());
            uint32_t num_pointee_children = CompilerType (getASTContext(),pointee_type).GetNumChildren (omit_empty_base_classes);
            if (num_pointee_children == 0)
            {
                // We have a pointer to a pointee type that claims it has no children.
                // We will want to look at
                num_children = GetNumPointeeChildren (pointee_type);
            }
            else
                num_children = num_pointee_children;
        }
            break;
            
        case clang::Type::LValueReference:
        case clang::Type::RValueReference:
        {
            const clang::ReferenceType *reference_type = llvm::cast<clang::ReferenceType>(qual_type.getTypePtr());
            clang::QualType pointee_type = reference_type->getPointeeType();
            uint32_t num_pointee_children = CompilerType (getASTContext(), pointee_type).GetNumChildren (omit_empty_base_classes);
            // If this type points to a simple type, then it has 1 child
            if (num_pointee_children == 0)
                num_children = 1;
            else
                num_children = num_pointee_children;
        }
            break;
            
            
        case clang::Type::Typedef:
            num_children = CompilerType (getASTContext(), llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType()).GetNumChildren (omit_empty_base_classes);
            break;

        case clang::Type::Auto:
            num_children = CompilerType (getASTContext(), llvm::cast<clang::AutoType>(qual_type)->getDeducedType()).GetNumChildren (omit_empty_base_classes);
            break;
            
        case clang::Type::Elaborated:
            num_children = CompilerType (getASTContext(), llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType()).GetNumChildren (omit_empty_base_classes);
            break;
            
        case clang::Type::Paren:
            num_children = CompilerType (getASTContext(), llvm::cast<clang::ParenType>(qual_type)->desugar()).GetNumChildren (omit_empty_base_classes);
            break;
        default:
            break;
    }
    return num_children;
}

CompilerType
ClangASTContext::GetBuiltinTypeByName (const ConstString &name)
{
    return GetBasicType (GetBasicTypeEnumeration (name));
}

lldb::BasicType
ClangASTContext::GetBasicTypeEnumeration (lldb::opaque_compiler_type_t type)
{
    if (type)
    {
        clang::QualType qual_type(GetQualType(type));
        const clang::Type::TypeClass type_class = qual_type->getTypeClass();
        if (type_class == clang::Type::Builtin)
        {
            switch (llvm::cast<clang::BuiltinType>(qual_type)->getKind())
            {
                case clang::BuiltinType::Void:      return eBasicTypeVoid;
                case clang::BuiltinType::Bool:      return eBasicTypeBool;
                case clang::BuiltinType::Char_S:    return eBasicTypeSignedChar;
                case clang::BuiltinType::Char_U:    return eBasicTypeUnsignedChar;
                case clang::BuiltinType::Char16:    return eBasicTypeChar16;
                case clang::BuiltinType::Char32:    return eBasicTypeChar32;
                case clang::BuiltinType::UChar:     return eBasicTypeUnsignedChar;
                case clang::BuiltinType::SChar:     return eBasicTypeSignedChar;
                case clang::BuiltinType::WChar_S:   return eBasicTypeSignedWChar;
                case clang::BuiltinType::WChar_U:   return eBasicTypeUnsignedWChar;
                case clang::BuiltinType::Short:     return eBasicTypeShort;
                case clang::BuiltinType::UShort:    return eBasicTypeUnsignedShort;
                case clang::BuiltinType::Int:       return eBasicTypeInt;
                case clang::BuiltinType::UInt:      return eBasicTypeUnsignedInt;
                case clang::BuiltinType::Long:      return eBasicTypeLong;
                case clang::BuiltinType::ULong:     return eBasicTypeUnsignedLong;
                case clang::BuiltinType::LongLong:  return eBasicTypeLongLong;
                case clang::BuiltinType::ULongLong: return eBasicTypeUnsignedLongLong;
                case clang::BuiltinType::Int128:    return eBasicTypeInt128;
                case clang::BuiltinType::UInt128:   return eBasicTypeUnsignedInt128;
                    
                case clang::BuiltinType::Half:      return eBasicTypeHalf;
                case clang::BuiltinType::Float:     return eBasicTypeFloat;
                case clang::BuiltinType::Double:    return eBasicTypeDouble;
                case clang::BuiltinType::LongDouble:return eBasicTypeLongDouble;
                    
                case clang::BuiltinType::NullPtr:   return eBasicTypeNullPtr;
                case clang::BuiltinType::ObjCId:    return eBasicTypeObjCID;
                case clang::BuiltinType::ObjCClass: return eBasicTypeObjCClass;
                case clang::BuiltinType::ObjCSel:   return eBasicTypeObjCSel;
                default:
                    return eBasicTypeOther;
            }
        }
    }
    return eBasicTypeInvalid;
}

void
ClangASTContext::ForEachEnumerator (lldb::opaque_compiler_type_t type, std::function <bool (const CompilerType &integer_type, const ConstString &name, const llvm::APSInt &value)> const &callback)
{
    const clang::EnumType *enum_type = llvm::dyn_cast<clang::EnumType>(GetCanonicalQualType(type));
    if (enum_type)
    {
        const clang::EnumDecl *enum_decl = enum_type->getDecl();
        if (enum_decl)
        {
            CompilerType integer_type(this, enum_decl->getIntegerType().getAsOpaquePtr());

            clang::EnumDecl::enumerator_iterator enum_pos, enum_end_pos;
            for (enum_pos = enum_decl->enumerator_begin(), enum_end_pos = enum_decl->enumerator_end(); enum_pos != enum_end_pos; ++enum_pos)
            {
                ConstString name(enum_pos->getNameAsString().c_str());
                if (!callback (integer_type, name, enum_pos->getInitVal()))
                    break;
            }
        }
    }
}


#pragma mark Aggregate Types

uint32_t
ClangASTContext::GetNumFields (lldb::opaque_compiler_type_t type)
{
    if (!type)
        return 0;
    
    uint32_t count = 0;
    clang::QualType qual_type(GetCanonicalQualType(type));
    const clang::Type::TypeClass type_class = qual_type->getTypeClass();
    switch (type_class)
    {
        case clang::Type::Record:
            if (GetCompleteType(type))
            {
                const clang::RecordType *record_type = llvm::dyn_cast<clang::RecordType>(qual_type.getTypePtr());
                if (record_type)
                {
                    clang::RecordDecl *record_decl = record_type->getDecl();
                    if (record_decl)
                    {
                        uint32_t field_idx = 0;
                        clang::RecordDecl::field_iterator field, field_end;
                        for (field = record_decl->field_begin(), field_end = record_decl->field_end(); field != field_end; ++field)
                            ++field_idx;
                        count = field_idx;
                    }
                }
            }
            break;
            
        case clang::Type::Typedef:
            count = CompilerType (getASTContext(), llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType()).GetNumFields();
            break;

        case clang::Type::Auto:
            count = CompilerType (getASTContext(), llvm::cast<clang::AutoType>(qual_type)->getDeducedType()).GetNumFields();
            break;
            
        case clang::Type::Elaborated:
            count = CompilerType (getASTContext(), llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType()).GetNumFields();
            break;
            
        case clang::Type::Paren:
            count = CompilerType (getASTContext(), llvm::cast<clang::ParenType>(qual_type)->desugar()).GetNumFields();
            break;
            
        case clang::Type::ObjCObjectPointer:
            if (GetCompleteType(type))
            {
                const clang::ObjCObjectPointerType *objc_class_type = qual_type->getAsObjCInterfacePointerType();
                if (objc_class_type)
                {
                    clang::ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterfaceDecl();
                    
                    if (class_interface_decl)
                        count = class_interface_decl->ivar_size();
                }
            }
            break;
            
        case clang::Type::ObjCObject:
        case clang::Type::ObjCInterface:
            if (GetCompleteType(type))
            {
                const clang::ObjCObjectType *objc_class_type = llvm::dyn_cast<clang::ObjCObjectType>(qual_type.getTypePtr());
                if (objc_class_type)
                {
                    clang::ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterface();
                    
                    if (class_interface_decl)
                        count = class_interface_decl->ivar_size();
                }
            }
            break;
            
        default:
            break;
    }
    return count;
}

static lldb::opaque_compiler_type_t
GetObjCFieldAtIndex (clang::ASTContext *ast,
                     clang::ObjCInterfaceDecl *class_interface_decl,
                     size_t idx,
                     std::string& name,
                     uint64_t *bit_offset_ptr,
                     uint32_t *bitfield_bit_size_ptr,
                     bool *is_bitfield_ptr)
{
    if (class_interface_decl)
    {
        if (idx < (class_interface_decl->ivar_size()))
        {
            clang::ObjCInterfaceDecl::ivar_iterator ivar_pos, ivar_end = class_interface_decl->ivar_end();
            uint32_t ivar_idx = 0;
            
            for (ivar_pos = class_interface_decl->ivar_begin(); ivar_pos != ivar_end; ++ivar_pos, ++ivar_idx)
            {
                if (ivar_idx == idx)
                {
                    const clang::ObjCIvarDecl* ivar_decl = *ivar_pos;
                    
                    clang::QualType ivar_qual_type(ivar_decl->getType());
                    
                    name.assign(ivar_decl->getNameAsString());
                    
                    if (bit_offset_ptr)
                    {
                        const clang::ASTRecordLayout &interface_layout = ast->getASTObjCInterfaceLayout(class_interface_decl);
                        *bit_offset_ptr = interface_layout.getFieldOffset (ivar_idx);
                    }
                    
                    const bool is_bitfield = ivar_pos->isBitField();
                    
                    if (bitfield_bit_size_ptr)
                    {
                        *bitfield_bit_size_ptr = 0;
                        
                        if (is_bitfield && ast)
                        {
                            clang::Expr *bitfield_bit_size_expr = ivar_pos->getBitWidth();
                            llvm::APSInt bitfield_apsint;
                            if (bitfield_bit_size_expr && bitfield_bit_size_expr->EvaluateAsInt(bitfield_apsint, *ast))
                            {
                                *bitfield_bit_size_ptr = bitfield_apsint.getLimitedValue();
                            }
                        }
                    }
                    if (is_bitfield_ptr)
                        *is_bitfield_ptr = is_bitfield;
                    
                    return ivar_qual_type.getAsOpaquePtr();
                }
            }
        }
    }
    return nullptr;
}

CompilerType
ClangASTContext::GetFieldAtIndex (lldb::opaque_compiler_type_t type, size_t idx,
                                     std::string& name,
                                     uint64_t *bit_offset_ptr,
                                     uint32_t *bitfield_bit_size_ptr,
                                     bool *is_bitfield_ptr)
{
    if (!type)
        return CompilerType();
    
    clang::QualType qual_type(GetCanonicalQualType(type));
    const clang::Type::TypeClass type_class = qual_type->getTypeClass();
    switch (type_class)
    {
        case clang::Type::Record:
            if (GetCompleteType(type))
            {
                const clang::RecordType *record_type = llvm::cast<clang::RecordType>(qual_type.getTypePtr());
                const clang::RecordDecl *record_decl = record_type->getDecl();
                uint32_t field_idx = 0;
                clang::RecordDecl::field_iterator field, field_end;
                for (field = record_decl->field_begin(), field_end = record_decl->field_end(); field != field_end; ++field, ++field_idx)
                {
                    if (idx == field_idx)
                    {
                        // Print the member type if requested
                        // Print the member name and equal sign
                        name.assign(field->getNameAsString());
                        
                        // Figure out the type byte size (field_type_info.first) and
                        // alignment (field_type_info.second) from the AST context.
                        if (bit_offset_ptr)
                        {
                            const clang::ASTRecordLayout &record_layout = getASTContext()->getASTRecordLayout(record_decl);
                            *bit_offset_ptr = record_layout.getFieldOffset (field_idx);
                        }
                        
                        const bool is_bitfield = field->isBitField();
                        
                        if (bitfield_bit_size_ptr)
                        {
                            *bitfield_bit_size_ptr = 0;
                            
                            if (is_bitfield)
                            {
                                clang::Expr *bitfield_bit_size_expr = field->getBitWidth();
                                llvm::APSInt bitfield_apsint;
                                if (bitfield_bit_size_expr && bitfield_bit_size_expr->EvaluateAsInt(bitfield_apsint, *getASTContext()))
                                {
                                    *bitfield_bit_size_ptr = bitfield_apsint.getLimitedValue();
                                }
                            }
                        }
                        if (is_bitfield_ptr)
                            *is_bitfield_ptr = is_bitfield;
                        
                        return CompilerType (getASTContext(), field->getType());
                    }
                }
            }
            break;
            
        case clang::Type::ObjCObjectPointer:
            if (GetCompleteType(type))
            {
                const clang::ObjCObjectPointerType *objc_class_type = qual_type->getAsObjCInterfacePointerType();
                if (objc_class_type)
                {
                    clang::ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterfaceDecl();
                    return CompilerType (this, GetObjCFieldAtIndex(getASTContext(), class_interface_decl, idx, name, bit_offset_ptr, bitfield_bit_size_ptr, is_bitfield_ptr));
                }
            }
            break;
            
        case clang::Type::ObjCObject:
        case clang::Type::ObjCInterface:
            if (GetCompleteType(type))
            {
                const clang::ObjCObjectType *objc_class_type = llvm::dyn_cast<clang::ObjCObjectType>(qual_type.getTypePtr());
                assert (objc_class_type);
                if (objc_class_type)
                {
                    clang::ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterface();
                    return CompilerType (this, GetObjCFieldAtIndex(getASTContext(), class_interface_decl, idx, name, bit_offset_ptr, bitfield_bit_size_ptr, is_bitfield_ptr));
                }
            }
            break;
            
            
        case clang::Type::Typedef:
            return CompilerType (getASTContext(), llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType()).
            GetFieldAtIndex (idx,
                             name,
                             bit_offset_ptr,
                             bitfield_bit_size_ptr,
                             is_bitfield_ptr);
            
        case clang::Type::Auto:
            return CompilerType (getASTContext(), llvm::cast<clang::AutoType>(qual_type)->getDeducedType()).
            GetFieldAtIndex (idx,
                             name,
                             bit_offset_ptr,
                             bitfield_bit_size_ptr,
                             is_bitfield_ptr);
            
        case clang::Type::Elaborated:
            return CompilerType (getASTContext(), llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType()).
            GetFieldAtIndex (idx,
                             name,
                             bit_offset_ptr,
                             bitfield_bit_size_ptr,
                             is_bitfield_ptr);
            
        case clang::Type::Paren:
            return CompilerType (getASTContext(), llvm::cast<clang::ParenType>(qual_type)->desugar()).
            GetFieldAtIndex (idx,
                             name,
                             bit_offset_ptr,
                             bitfield_bit_size_ptr,
                             is_bitfield_ptr);
            
        default:
            break;
    }
    return CompilerType();
}

uint32_t
ClangASTContext::GetNumDirectBaseClasses (lldb::opaque_compiler_type_t type)
{
    uint32_t count = 0;
    clang::QualType qual_type(GetCanonicalQualType(type));
    const clang::Type::TypeClass type_class = qual_type->getTypeClass();
    switch (type_class)
    {
        case clang::Type::Record:
            if (GetCompleteType(type))
            {
                const clang::CXXRecordDecl *cxx_record_decl = qual_type->getAsCXXRecordDecl();
                if (cxx_record_decl)
                    count = cxx_record_decl->getNumBases();
            }
            break;

        case clang::Type::ObjCObjectPointer:
            count = GetPointeeType(type).GetNumDirectBaseClasses();
            break;

        case clang::Type::ObjCObject:
            if (GetCompleteType(type))
            {
                const clang::ObjCObjectType *objc_class_type = qual_type->getAsObjCQualifiedInterfaceType();
                if (objc_class_type)
                {
                    clang::ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterface();

                    if (class_interface_decl && class_interface_decl->getSuperClass())
                        count = 1;
                }
            }
            break;
        case clang::Type::ObjCInterface:
            if (GetCompleteType(type))
            {
                const clang::ObjCInterfaceType *objc_interface_type = qual_type->getAs<clang::ObjCInterfaceType>();
                if (objc_interface_type)
                {
                    clang::ObjCInterfaceDecl *class_interface_decl = objc_interface_type->getInterface();

                    if (class_interface_decl && class_interface_decl->getSuperClass())
                        count = 1;
                }
            }
            break;


        case clang::Type::Typedef:
            count = GetNumDirectBaseClasses(llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType().getAsOpaquePtr());
            break;

        case clang::Type::Auto:
            count = GetNumDirectBaseClasses(llvm::cast<clang::AutoType>(qual_type)->getDeducedType().getAsOpaquePtr());
            break;
            
        case clang::Type::Elaborated:
            count = GetNumDirectBaseClasses(llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType().getAsOpaquePtr());
            break;

        case clang::Type::Paren:
            return GetNumDirectBaseClasses(llvm::cast<clang::ParenType>(qual_type)->desugar().getAsOpaquePtr());

        default:
            break;
    }
    return count;

}

uint32_t
ClangASTContext::GetNumVirtualBaseClasses (lldb::opaque_compiler_type_t type)
{
    uint32_t count = 0;
    clang::QualType qual_type(GetCanonicalQualType(type));
    const clang::Type::TypeClass type_class = qual_type->getTypeClass();
    switch (type_class)
    {
        case clang::Type::Record:
            if (GetCompleteType(type))
            {
                const clang::CXXRecordDecl *cxx_record_decl = qual_type->getAsCXXRecordDecl();
                if (cxx_record_decl)
                    count = cxx_record_decl->getNumVBases();
            }
            break;

        case clang::Type::Typedef:
            count = GetNumVirtualBaseClasses(llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType().getAsOpaquePtr());
            break;

        case clang::Type::Auto:
            count = GetNumVirtualBaseClasses(llvm::cast<clang::AutoType>(qual_type)->getDeducedType().getAsOpaquePtr());
            break;
            
        case clang::Type::Elaborated:
            count = GetNumVirtualBaseClasses(llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType().getAsOpaquePtr());
            break;

        case clang::Type::Paren:
            count = GetNumVirtualBaseClasses(llvm::cast<clang::ParenType>(qual_type)->desugar().getAsOpaquePtr());
            break;

        default:
            break;
    }
    return count;

}

CompilerType
ClangASTContext::GetDirectBaseClassAtIndex (lldb::opaque_compiler_type_t type, size_t idx, uint32_t *bit_offset_ptr)
{
    clang::QualType qual_type(GetCanonicalQualType(type));
    const clang::Type::TypeClass type_class = qual_type->getTypeClass();
    switch (type_class)
    {
        case clang::Type::Record:
            if (GetCompleteType(type))
            {
                const clang::CXXRecordDecl *cxx_record_decl = qual_type->getAsCXXRecordDecl();
                if (cxx_record_decl)
                {
                    uint32_t curr_idx = 0;
                    clang::CXXRecordDecl::base_class_const_iterator base_class, base_class_end;
                    for (base_class = cxx_record_decl->bases_begin(), base_class_end = cxx_record_decl->bases_end();
                         base_class != base_class_end;
                         ++base_class, ++curr_idx)
                    {
                        if (curr_idx == idx)
                        {
                            if (bit_offset_ptr)
                            {
                                const clang::ASTRecordLayout &record_layout = getASTContext()->getASTRecordLayout(cxx_record_decl);
                                const clang::CXXRecordDecl *base_class_decl = llvm::cast<clang::CXXRecordDecl>(base_class->getType()->getAs<clang::RecordType>()->getDecl());
                                if (base_class->isVirtual())
                                    *bit_offset_ptr = record_layout.getVBaseClassOffset(base_class_decl).getQuantity() * 8;
                                else
                                    *bit_offset_ptr = record_layout.getBaseClassOffset(base_class_decl).getQuantity() * 8;
                            }
                            return CompilerType (this, base_class->getType().getAsOpaquePtr());
                        }
                    }
                }
            }
            break;

        case clang::Type::ObjCObjectPointer:
            return GetPointeeType(type).GetDirectBaseClassAtIndex(idx, bit_offset_ptr);

        case clang::Type::ObjCObject:
            if (idx == 0 && GetCompleteType(type))
            {
                const clang::ObjCObjectType *objc_class_type = qual_type->getAsObjCQualifiedInterfaceType();
                if (objc_class_type)
                {
                    clang::ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterface();

                    if (class_interface_decl)
                    {
                        clang::ObjCInterfaceDecl *superclass_interface_decl = class_interface_decl->getSuperClass();
                        if (superclass_interface_decl)
                        {
                            if (bit_offset_ptr)
                                *bit_offset_ptr = 0;
                            return CompilerType (getASTContext(), getASTContext()->getObjCInterfaceType(superclass_interface_decl));
                        }
                    }
                }
            }
            break;
        case clang::Type::ObjCInterface:
            if (idx == 0 && GetCompleteType(type))
            {
                const clang::ObjCObjectType *objc_interface_type = qual_type->getAs<clang::ObjCInterfaceType>();
                if (objc_interface_type)
                {
                    clang::ObjCInterfaceDecl *class_interface_decl = objc_interface_type->getInterface();

                    if (class_interface_decl)
                    {
                        clang::ObjCInterfaceDecl *superclass_interface_decl = class_interface_decl->getSuperClass();
                        if (superclass_interface_decl)
                        {
                            if (bit_offset_ptr)
                                *bit_offset_ptr = 0;
                            return CompilerType (getASTContext(), getASTContext()->getObjCInterfaceType(superclass_interface_decl));
                        }
                    }
                }
            }
            break;


        case clang::Type::Typedef:
            return GetDirectBaseClassAtIndex (llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType().getAsOpaquePtr(), idx, bit_offset_ptr);

        case clang::Type::Auto:
            return GetDirectBaseClassAtIndex (llvm::cast<clang::AutoType>(qual_type)->getDeducedType().getAsOpaquePtr(), idx, bit_offset_ptr);
            
        case clang::Type::Elaborated:
            return GetDirectBaseClassAtIndex (llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType().getAsOpaquePtr(), idx, bit_offset_ptr);

        case clang::Type::Paren:
            return GetDirectBaseClassAtIndex (llvm::cast<clang::ParenType>(qual_type)->desugar().getAsOpaquePtr(), idx, bit_offset_ptr);

        default:
            break;
    }
    return CompilerType();
}

CompilerType
ClangASTContext::GetVirtualBaseClassAtIndex (lldb::opaque_compiler_type_t type,
                                             size_t idx,
                                             uint32_t *bit_offset_ptr)
{
    clang::QualType qual_type(GetCanonicalQualType(type));
    const clang::Type::TypeClass type_class = qual_type->getTypeClass();
    switch (type_class)
    {
        case clang::Type::Record:
            if (GetCompleteType(type))
            {
                const clang::CXXRecordDecl *cxx_record_decl = qual_type->getAsCXXRecordDecl();
                if (cxx_record_decl)
                {
                    uint32_t curr_idx = 0;
                    clang::CXXRecordDecl::base_class_const_iterator base_class, base_class_end;
                    for (base_class = cxx_record_decl->vbases_begin(), base_class_end = cxx_record_decl->vbases_end();
                         base_class != base_class_end;
                         ++base_class, ++curr_idx)
                    {
                        if (curr_idx == idx)
                        {
                            if (bit_offset_ptr)
                            {
                                const clang::ASTRecordLayout &record_layout = getASTContext()->getASTRecordLayout(cxx_record_decl);
                                const clang::CXXRecordDecl *base_class_decl = llvm::cast<clang::CXXRecordDecl>(base_class->getType()->getAs<clang::RecordType>()->getDecl());
                                *bit_offset_ptr = record_layout.getVBaseClassOffset(base_class_decl).getQuantity() * 8;

                            }
                            return CompilerType (this, base_class->getType().getAsOpaquePtr());
                        }
                    }
                }
            }
            break;

        case clang::Type::Typedef:
            return GetVirtualBaseClassAtIndex (llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType().getAsOpaquePtr(), idx, bit_offset_ptr);
            
        case clang::Type::Auto:
            return GetVirtualBaseClassAtIndex (llvm::cast<clang::AutoType>(qual_type)->getDeducedType().getAsOpaquePtr(), idx, bit_offset_ptr);
            
        case clang::Type::Elaborated:
            return GetVirtualBaseClassAtIndex (llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType().getAsOpaquePtr(), idx, bit_offset_ptr);

        case clang::Type::Paren:
            return  GetVirtualBaseClassAtIndex (llvm::cast<clang::ParenType>(qual_type)->desugar().getAsOpaquePtr(), idx, bit_offset_ptr);

        default:
            break;
    }
    return CompilerType();

}

// If a pointer to a pointee type (the clang_type arg) says that it has no
// children, then we either need to trust it, or override it and return a
// different result. For example, an "int *" has one child that is an integer,
// but a function pointer doesn't have any children. Likewise if a Record type
// claims it has no children, then there really is nothing to show.
uint32_t
ClangASTContext::GetNumPointeeChildren (clang::QualType type)
{
    if (type.isNull())
        return 0;
    
    clang::QualType qual_type(type.getCanonicalType());
    const clang::Type::TypeClass type_class = qual_type->getTypeClass();
    switch (type_class)
    {
        case clang::Type::Builtin:
            switch (llvm::cast<clang::BuiltinType>(qual_type)->getKind())
        {
            case clang::BuiltinType::UnknownAny:
            case clang::BuiltinType::Void:
            case clang::BuiltinType::NullPtr:
            case clang::BuiltinType::OCLEvent:
            case clang::BuiltinType::OCLImage1d:
            case clang::BuiltinType::OCLImage1dArray:
            case clang::BuiltinType::OCLImage1dBuffer:
            case clang::BuiltinType::OCLImage2d:
            case clang::BuiltinType::OCLImage2dArray:
            case clang::BuiltinType::OCLImage3d:
            case clang::BuiltinType::OCLSampler:
                return 0;
            case clang::BuiltinType::Bool:
            case clang::BuiltinType::Char_U:
            case clang::BuiltinType::UChar:
            case clang::BuiltinType::WChar_U:
            case clang::BuiltinType::Char16:
            case clang::BuiltinType::Char32:
            case clang::BuiltinType::UShort:
            case clang::BuiltinType::UInt:
            case clang::BuiltinType::ULong:
            case clang::BuiltinType::ULongLong:
            case clang::BuiltinType::UInt128:
            case clang::BuiltinType::Char_S:
            case clang::BuiltinType::SChar:
            case clang::BuiltinType::WChar_S:
            case clang::BuiltinType::Short:
            case clang::BuiltinType::Int:
            case clang::BuiltinType::Long:
            case clang::BuiltinType::LongLong:
            case clang::BuiltinType::Int128:
            case clang::BuiltinType::Float:
            case clang::BuiltinType::Double:
            case clang::BuiltinType::LongDouble:
            case clang::BuiltinType::Dependent:
            case clang::BuiltinType::Overload:
            case clang::BuiltinType::ObjCId:
            case clang::BuiltinType::ObjCClass:
            case clang::BuiltinType::ObjCSel:
            case clang::BuiltinType::BoundMember:
            case clang::BuiltinType::Half:
            case clang::BuiltinType::ARCUnbridgedCast:
            case clang::BuiltinType::PseudoObject:
            case clang::BuiltinType::BuiltinFn:
            case clang::BuiltinType::OMPArraySection:
                return 1;
            default:
                return 0;
        }
            break;
            
        case clang::Type::Complex:                  return 1;
        case clang::Type::Pointer:                  return 1;
        case clang::Type::BlockPointer:             return 0;   // If block pointers don't have debug info, then no children for them
        case clang::Type::LValueReference:          return 1;
        case clang::Type::RValueReference:          return 1;
        case clang::Type::MemberPointer:            return 0;
        case clang::Type::ConstantArray:            return 0;
        case clang::Type::IncompleteArray:          return 0;
        case clang::Type::VariableArray:            return 0;
        case clang::Type::DependentSizedArray:      return 0;
        case clang::Type::DependentSizedExtVector:  return 0;
        case clang::Type::Vector:                   return 0;
        case clang::Type::ExtVector:                return 0;
        case clang::Type::FunctionProto:            return 0;   // When we function pointers, they have no children...
        case clang::Type::FunctionNoProto:          return 0;   // When we function pointers, they have no children...
        case clang::Type::UnresolvedUsing:          return 0;
        case clang::Type::Paren:                    return GetNumPointeeChildren (llvm::cast<clang::ParenType>(qual_type)->desugar());
        case clang::Type::Typedef:                  return GetNumPointeeChildren (llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType());
        case clang::Type::Auto:                     return GetNumPointeeChildren (llvm::cast<clang::AutoType>(qual_type)->getDeducedType());
        case clang::Type::Elaborated:               return GetNumPointeeChildren (llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType());
        case clang::Type::TypeOfExpr:               return 0;
        case clang::Type::TypeOf:                   return 0;
        case clang::Type::Decltype:                 return 0;
        case clang::Type::Record:                   return 0;
        case clang::Type::Enum:                     return 1;
        case clang::Type::TemplateTypeParm:         return 1;
        case clang::Type::SubstTemplateTypeParm:    return 1;
        case clang::Type::TemplateSpecialization:   return 1;
        case clang::Type::InjectedClassName:        return 0;
        case clang::Type::DependentName:            return 1;
        case clang::Type::DependentTemplateSpecialization:  return 1;
        case clang::Type::ObjCObject:               return 0;
        case clang::Type::ObjCInterface:            return 0;
        case clang::Type::ObjCObjectPointer:        return 1;
        default:
            break;
    }
    return 0;
}


CompilerType
ClangASTContext::GetChildCompilerTypeAtIndex (lldb::opaque_compiler_type_t type,
                                              ExecutionContext *exe_ctx,
                                              size_t idx,
                                              bool transparent_pointers,
                                              bool omit_empty_base_classes,
                                              bool ignore_array_bounds,
                                              std::string& child_name,
                                              uint32_t &child_byte_size,
                                              int32_t &child_byte_offset,
                                              uint32_t &child_bitfield_bit_size,
                                              uint32_t &child_bitfield_bit_offset,
                                              bool &child_is_base_class,
                                              bool &child_is_deref_of_parent,
                                              ValueObject *valobj,
                                              uint64_t &language_flags)
{
    if (!type)
        return CompilerType();
    
    clang::QualType parent_qual_type(GetCanonicalQualType(type));
    const clang::Type::TypeClass parent_type_class = parent_qual_type->getTypeClass();
    child_bitfield_bit_size = 0;
    child_bitfield_bit_offset = 0;
    child_is_base_class = false;
    language_flags = 0;
    
    const bool idx_is_valid = idx < GetNumChildren (type, omit_empty_base_classes);
    uint32_t bit_offset;
    switch (parent_type_class)
    {
        case clang::Type::Builtin:
            if (idx_is_valid)
            {
                switch (llvm::cast<clang::BuiltinType>(parent_qual_type)->getKind())
                {
                    case clang::BuiltinType::ObjCId:
                    case clang::BuiltinType::ObjCClass:
                        child_name = "isa";
                        child_byte_size = getASTContext()->getTypeSize(getASTContext()->ObjCBuiltinClassTy) / CHAR_BIT;
                        return CompilerType (getASTContext(), getASTContext()->ObjCBuiltinClassTy);
                        
                    default:
                        break;
                }
            }
            break;
            
        case clang::Type::Record:
            if (idx_is_valid && GetCompleteType(type))
            {
                const clang::RecordType *record_type = llvm::cast<clang::RecordType>(parent_qual_type.getTypePtr());
                const clang::RecordDecl *record_decl = record_type->getDecl();
                assert(record_decl);
                const clang::ASTRecordLayout &record_layout = getASTContext()->getASTRecordLayout(record_decl);
                uint32_t child_idx = 0;
                
                const clang::CXXRecordDecl *cxx_record_decl = llvm::dyn_cast<clang::CXXRecordDecl>(record_decl);
                if (cxx_record_decl)
                {
                    // We might have base classes to print out first
                    clang::CXXRecordDecl::base_class_const_iterator base_class, base_class_end;
                    for (base_class = cxx_record_decl->bases_begin(), base_class_end = cxx_record_decl->bases_end();
                         base_class != base_class_end;
                         ++base_class)
                    {
                        const clang::CXXRecordDecl *base_class_decl = nullptr;
                        
                        // Skip empty base classes
                        if (omit_empty_base_classes)
                        {
                            base_class_decl = llvm::cast<clang::CXXRecordDecl>(base_class->getType()->getAs<clang::RecordType>()->getDecl());
                            if (ClangASTContext::RecordHasFields(base_class_decl) == false)
                                continue;
                        }
                        
                        if (idx == child_idx)
                        {
                            if (base_class_decl == nullptr)
                                base_class_decl = llvm::cast<clang::CXXRecordDecl>(base_class->getType()->getAs<clang::RecordType>()->getDecl());
                            
                            
                            if (base_class->isVirtual())
                            {
                                bool handled = false;
                                if (valobj)
                                {
                                    Error err;
                                    AddressType addr_type = eAddressTypeInvalid;
                                    lldb::addr_t vtable_ptr_addr = valobj->GetCPPVTableAddress(addr_type);
                                    
                                    if (vtable_ptr_addr != LLDB_INVALID_ADDRESS && addr_type == eAddressTypeLoad)
                                    {
                                        
                                        ExecutionContext exe_ctx (valobj->GetExecutionContextRef());
                                        Process *process = exe_ctx.GetProcessPtr();
                                        if (process)
                                        {
                                            clang::VTableContextBase *vtable_ctx = getASTContext()->getVTableContext();
                                            if (vtable_ctx)
                                            {
                                                if (vtable_ctx->isMicrosoft())
                                                {
                                                    clang::MicrosoftVTableContext *msoft_vtable_ctx = static_cast<clang::MicrosoftVTableContext *>(vtable_ctx);
                                                    
                                                    if (vtable_ptr_addr)
                                                    {
                                                        const lldb::addr_t vbtable_ptr_addr = vtable_ptr_addr + record_layout.getVBPtrOffset().getQuantity();
                                                        
                                                        const lldb::addr_t vbtable_ptr = process->ReadPointerFromMemory(vbtable_ptr_addr, err);
                                                        if (vbtable_ptr != LLDB_INVALID_ADDRESS)
                                                        {
                                                            // Get the index into the virtual base table. The index is the index in uint32_t from vbtable_ptr
                                                            const unsigned vbtable_index = msoft_vtable_ctx->getVBTableIndex(cxx_record_decl, base_class_decl);
                                                            const lldb::addr_t base_offset_addr = vbtable_ptr + vbtable_index * 4;
                                                            const uint32_t base_offset = process->ReadUnsignedIntegerFromMemory(base_offset_addr, 4, UINT32_MAX, err);
                                                            if (base_offset != UINT32_MAX)
                                                            {
                                                                handled = true;
                                                                bit_offset = base_offset * 8;
                                                            }
                                                        }
                                                    }
                                                }
                                                else
                                                {
                                                    clang::ItaniumVTableContext *itanium_vtable_ctx = static_cast<clang::ItaniumVTableContext *>(vtable_ctx);
                                                    if (vtable_ptr_addr)
                                                    {
                                                        const lldb::addr_t vtable_ptr = process->ReadPointerFromMemory(vtable_ptr_addr, err);
                                                        if (vtable_ptr != LLDB_INVALID_ADDRESS)
                                                        {
                                                            clang::CharUnits base_offset_offset = itanium_vtable_ctx->getVirtualBaseOffsetOffset(cxx_record_decl, base_class_decl);
                                                            const lldb::addr_t base_offset_addr = vtable_ptr + base_offset_offset.getQuantity();
                                                            const uint32_t base_offset = process->ReadUnsignedIntegerFromMemory(base_offset_addr, 4, UINT32_MAX, err);
                                                            if (base_offset != UINT32_MAX)
                                                            {
                                                                handled = true;
                                                                bit_offset = base_offset * 8;
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    
                                }
                                if (!handled)
                                    bit_offset = record_layout.getVBaseClassOffset(base_class_decl).getQuantity() * 8;
                            }
                            else
                                bit_offset = record_layout.getBaseClassOffset(base_class_decl).getQuantity() * 8;
                            
                            // Base classes should be a multiple of 8 bits in size
                            child_byte_offset = bit_offset/8;
                            CompilerType base_class_clang_type(getASTContext(), base_class->getType());
                            child_name = base_class_clang_type.GetTypeName().AsCString("");
                            uint64_t base_class_clang_type_bit_size = base_class_clang_type.GetBitSize(exe_ctx ? exe_ctx->GetBestExecutionContextScope() : NULL);
                            
                            // Base classes bit sizes should be a multiple of 8 bits in size
                            assert (base_class_clang_type_bit_size % 8 == 0);
                            child_byte_size = base_class_clang_type_bit_size / 8;
                            child_is_base_class = true;
                            return base_class_clang_type;
                        }
                        // We don't increment the child index in the for loop since we might
                        // be skipping empty base classes
                        ++child_idx;
                    }
                }
                // Make sure index is in range...
                uint32_t field_idx = 0;
                clang::RecordDecl::field_iterator field, field_end;
                for (field = record_decl->field_begin(), field_end = record_decl->field_end(); field != field_end; ++field, ++field_idx, ++child_idx)
                {
                    if (idx == child_idx)
                    {
                        // Print the member type if requested
                        // Print the member name and equal sign
                        child_name.assign(field->getNameAsString().c_str());
                        
                        // Figure out the type byte size (field_type_info.first) and
                        // alignment (field_type_info.second) from the AST context.
                        CompilerType field_clang_type (getASTContext(), field->getType());
                        assert(field_idx < record_layout.getFieldCount());
                        child_byte_size = field_clang_type.GetByteSize(exe_ctx ? exe_ctx->GetBestExecutionContextScope() : NULL);
                        
                        // Figure out the field offset within the current struct/union/class type
                        bit_offset = record_layout.getFieldOffset (field_idx);
                        child_byte_offset = bit_offset / 8;
                        if (ClangASTContext::FieldIsBitfield (getASTContext(), *field, child_bitfield_bit_size))
                            child_bitfield_bit_offset = bit_offset % 8;
                        
                        return field_clang_type;
                    }
                }
            }
            break;
            
        case clang::Type::ObjCObject:
        case clang::Type::ObjCInterface:
            if (idx_is_valid && GetCompleteType(type))
            {
                const clang::ObjCObjectType *objc_class_type = llvm::dyn_cast<clang::ObjCObjectType>(parent_qual_type.getTypePtr());
                assert (objc_class_type);
                if (objc_class_type)
                {
                    uint32_t child_idx = 0;
                    clang::ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterface();
                    
                    if (class_interface_decl)
                    {
                        
                        const clang::ASTRecordLayout &interface_layout = getASTContext()->getASTObjCInterfaceLayout(class_interface_decl);
                        clang::ObjCInterfaceDecl *superclass_interface_decl = class_interface_decl->getSuperClass();
                        if (superclass_interface_decl)
                        {
                            if (omit_empty_base_classes)
                            {
                                CompilerType base_class_clang_type (getASTContext(), getASTContext()->getObjCInterfaceType(superclass_interface_decl));
                                if (base_class_clang_type.GetNumChildren(omit_empty_base_classes) > 0)
                                {
                                    if (idx == 0)
                                    {
                                        clang::QualType ivar_qual_type(getASTContext()->getObjCInterfaceType(superclass_interface_decl));
                                        
                                        
                                        child_name.assign(superclass_interface_decl->getNameAsString().c_str());
                                        
                                        clang::TypeInfo ivar_type_info = getASTContext()->getTypeInfo(ivar_qual_type.getTypePtr());
                                        
                                        child_byte_size = ivar_type_info.Width / 8;
                                        child_byte_offset = 0;
                                        child_is_base_class = true;
                                        
                                        return CompilerType (getASTContext(), ivar_qual_type);
                                    }
                                    
                                    ++child_idx;
                                }
                            }
                            else
                                ++child_idx;
                        }
                        
                        const uint32_t superclass_idx = child_idx;
                        
                        if (idx < (child_idx + class_interface_decl->ivar_size()))
                        {
                            clang::ObjCInterfaceDecl::ivar_iterator ivar_pos, ivar_end = class_interface_decl->ivar_end();
                            
                            for (ivar_pos = class_interface_decl->ivar_begin(); ivar_pos != ivar_end; ++ivar_pos)
                            {
                                if (child_idx == idx)
                                {
                                    clang::ObjCIvarDecl* ivar_decl = *ivar_pos;
                                    
                                    clang::QualType ivar_qual_type(ivar_decl->getType());
                                    
                                    child_name.assign(ivar_decl->getNameAsString().c_str());
                                    
                                    clang::TypeInfo  ivar_type_info = getASTContext()->getTypeInfo(ivar_qual_type.getTypePtr());
                                    
                                    child_byte_size = ivar_type_info.Width / 8;
                                    
                                    // Figure out the field offset within the current struct/union/class type
                                    // For ObjC objects, we can't trust the bit offset we get from the Clang AST, since
                                    // that doesn't account for the space taken up by unbacked properties, or from
                                    // the changing size of base classes that are newer than this class.
                                    // So if we have a process around that we can ask about this object, do so.
                                    child_byte_offset = LLDB_INVALID_IVAR_OFFSET;
                                    Process *process = nullptr;
                                    if (exe_ctx)
                                        process = exe_ctx->GetProcessPtr();
                                    if (process)
                                    {
                                        ObjCLanguageRuntime *objc_runtime = process->GetObjCLanguageRuntime();
                                        if (objc_runtime != nullptr)
                                        {
                                            CompilerType parent_ast_type (getASTContext(), parent_qual_type);
                                            child_byte_offset = objc_runtime->GetByteOffsetForIvar (parent_ast_type, ivar_decl->getNameAsString().c_str());
                                        }
                                    }
                                    
                                    // Setting this to UINT32_MAX to make sure we don't compute it twice...
                                    bit_offset = UINT32_MAX;
                                    
                                    if (child_byte_offset == static_cast<int32_t>(LLDB_INVALID_IVAR_OFFSET))
                                    {
                                        bit_offset = interface_layout.getFieldOffset (child_idx - superclass_idx);
                                        child_byte_offset = bit_offset / 8;
                                    }
                                    
                                    // Note, the ObjC Ivar Byte offset is just that, it doesn't account for the bit offset
                                    // of a bitfield within its containing object.  So regardless of where we get the byte
                                    // offset from, we still need to get the bit offset for bitfields from the layout.
                                    
                                    if (ClangASTContext::FieldIsBitfield (getASTContext(), ivar_decl, child_bitfield_bit_size))
                                    {
                                        if (bit_offset == UINT32_MAX)
                                            bit_offset = interface_layout.getFieldOffset (child_idx - superclass_idx);
                                        
                                        child_bitfield_bit_offset = bit_offset % 8;
                                    }
                                    return CompilerType (getASTContext(), ivar_qual_type);
                                }
                                ++child_idx;
                            }
                        }
                    }
                }
            }
            break;
            
        case clang::Type::ObjCObjectPointer:
            if (idx_is_valid)
            {
                CompilerType pointee_clang_type (GetPointeeType(type));
                
                if (transparent_pointers && pointee_clang_type.IsAggregateType())
                {
                    child_is_deref_of_parent = false;
                    bool tmp_child_is_deref_of_parent = false;
                    return pointee_clang_type.GetChildCompilerTypeAtIndex (exe_ctx,
                                                                           idx,
                                                                           transparent_pointers,
                                                                           omit_empty_base_classes,
                                                                           ignore_array_bounds,
                                                                           child_name,
                                                                           child_byte_size,
                                                                           child_byte_offset,
                                                                           child_bitfield_bit_size,
                                                                           child_bitfield_bit_offset,
                                                                           child_is_base_class,
                                                                           tmp_child_is_deref_of_parent,
                                                                           valobj,
                                                                           language_flags);
                }
                else
                {
                    child_is_deref_of_parent = true;
                    const char *parent_name = valobj ? valobj->GetName().GetCString() : NULL;
                    if (parent_name)
                    {
                        child_name.assign(1, '*');
                        child_name += parent_name;
                    }
                    
                    // We have a pointer to an simple type
                    if (idx == 0 && pointee_clang_type.GetCompleteType())
                    {
                        child_byte_size = pointee_clang_type.GetByteSize(exe_ctx ? exe_ctx->GetBestExecutionContextScope() : NULL);
                        child_byte_offset = 0;
                        return pointee_clang_type;
                    }
                }
            }
            break;
            
        case clang::Type::Vector:
        case clang::Type::ExtVector:
            if (idx_is_valid)
            {
                const clang::VectorType *array = llvm::cast<clang::VectorType>(parent_qual_type.getTypePtr());
                if (array)
                {
                    CompilerType element_type (getASTContext(), array->getElementType());
                    if (element_type.GetCompleteType())
                    {
                        char element_name[64];
                        ::snprintf (element_name, sizeof (element_name), "[%" PRIu64 "]", static_cast<uint64_t>(idx));
                        child_name.assign(element_name);
                        child_byte_size = element_type.GetByteSize(exe_ctx ? exe_ctx->GetBestExecutionContextScope() : NULL);
                        child_byte_offset = (int32_t)idx * (int32_t)child_byte_size;
                        return element_type;
                    }
                }
            }
            break;
            
        case clang::Type::ConstantArray:
        case clang::Type::IncompleteArray:
            if (ignore_array_bounds || idx_is_valid)
            {
                const clang::ArrayType *array = GetQualType(type)->getAsArrayTypeUnsafe();
                if (array)
                {
                    CompilerType element_type (getASTContext(), array->getElementType());
                    if (element_type.GetCompleteType())
                    {
                        char element_name[64];
                        ::snprintf (element_name, sizeof (element_name), "[%" PRIu64 "]", static_cast<uint64_t>(idx));
                        child_name.assign(element_name);
                        child_byte_size = element_type.GetByteSize(exe_ctx ? exe_ctx->GetBestExecutionContextScope() : NULL);
                        child_byte_offset = (int32_t)idx * (int32_t)child_byte_size;
                        return element_type;
                    }
                }
            }
            break;
            
            
        case clang::Type::Pointer:
            if (idx_is_valid)
            {
                CompilerType pointee_clang_type (GetPointeeType(type));
                
                // Don't dereference "void *" pointers
                if (pointee_clang_type.IsVoidType())
                    return CompilerType();
                
                if (transparent_pointers && pointee_clang_type.IsAggregateType ())
                {
                    child_is_deref_of_parent = false;
                    bool tmp_child_is_deref_of_parent = false;
                    return pointee_clang_type.GetChildCompilerTypeAtIndex (exe_ctx,
                                                                           idx,
                                                                           transparent_pointers,
                                                                           omit_empty_base_classes,
                                                                           ignore_array_bounds,
                                                                           child_name,
                                                                           child_byte_size,
                                                                           child_byte_offset,
                                                                           child_bitfield_bit_size,
                                                                           child_bitfield_bit_offset,
                                                                           child_is_base_class,
                                                                           tmp_child_is_deref_of_parent,
                                                                           valobj,
                                                                           language_flags);
                }
                else
                {
                    child_is_deref_of_parent = true;
                    
                    const char *parent_name = valobj ? valobj->GetName().GetCString() : NULL;
                    if (parent_name)
                    {
                        child_name.assign(1, '*');
                        child_name += parent_name;
                    }
                    
                    // We have a pointer to an simple type
                    if (idx == 0)
                    {
                        child_byte_size = pointee_clang_type.GetByteSize(exe_ctx ? exe_ctx->GetBestExecutionContextScope() : NULL);
                        child_byte_offset = 0;
                        return pointee_clang_type;
                    }
                }
            }
            break;
            
        case clang::Type::LValueReference:
        case clang::Type::RValueReference:
            if (idx_is_valid)
            {
                const clang::ReferenceType *reference_type = llvm::cast<clang::ReferenceType>(parent_qual_type.getTypePtr());
                CompilerType pointee_clang_type (getASTContext(), reference_type->getPointeeType());
                if (transparent_pointers && pointee_clang_type.IsAggregateType ())
                {
                    child_is_deref_of_parent = false;
                    bool tmp_child_is_deref_of_parent = false;
                    return pointee_clang_type.GetChildCompilerTypeAtIndex (exe_ctx,
                                                                           idx,
                                                                           transparent_pointers,
                                                                           omit_empty_base_classes,
                                                                           ignore_array_bounds,
                                                                           child_name,
                                                                           child_byte_size,
                                                                           child_byte_offset,
                                                                           child_bitfield_bit_size,
                                                                           child_bitfield_bit_offset,
                                                                           child_is_base_class,
                                                                           tmp_child_is_deref_of_parent,
                                                                           valobj,
                                                                           language_flags);
                }
                else
                {
                    const char *parent_name = valobj ? valobj->GetName().GetCString() : NULL;
                    if (parent_name)
                    {
                        child_name.assign(1, '&');
                        child_name += parent_name;
                    }
                    
                    // We have a pointer to an simple type
                    if (idx == 0)
                    {
                        child_byte_size = pointee_clang_type.GetByteSize(exe_ctx ? exe_ctx->GetBestExecutionContextScope() : NULL);
                        child_byte_offset = 0;
                        return pointee_clang_type;
                    }
                }
            }
            break;
            
        case clang::Type::Typedef:
        {
            CompilerType typedefed_clang_type (getASTContext(), llvm::cast<clang::TypedefType>(parent_qual_type)->getDecl()->getUnderlyingType());
            return typedefed_clang_type.GetChildCompilerTypeAtIndex (exe_ctx,
                                                                     idx,
                                                                     transparent_pointers,
                                                                     omit_empty_base_classes,
                                                                     ignore_array_bounds,
                                                                     child_name,
                                                                     child_byte_size,
                                                                     child_byte_offset,
                                                                     child_bitfield_bit_size,
                                                                     child_bitfield_bit_offset,
                                                                     child_is_base_class,
                                                                     child_is_deref_of_parent,
                                                                     valobj,
                                                                     language_flags);
        }
            break;
            
        case clang::Type::Auto:
        {
            CompilerType elaborated_clang_type (getASTContext(), llvm::cast<clang::AutoType>(parent_qual_type)->getDeducedType());
            return elaborated_clang_type.GetChildCompilerTypeAtIndex (exe_ctx,
                                                                      idx,
                                                                      transparent_pointers,
                                                                      omit_empty_base_classes,
                                                                      ignore_array_bounds,
                                                                      child_name,
                                                                      child_byte_size,
                                                                      child_byte_offset,
                                                                      child_bitfield_bit_size,
                                                                      child_bitfield_bit_offset,
                                                                      child_is_base_class,
                                                                      child_is_deref_of_parent,
                                                                      valobj,
                                                                      language_flags);
        }
            
        case clang::Type::Elaborated:
        {
            CompilerType elaborated_clang_type (getASTContext(), llvm::cast<clang::ElaboratedType>(parent_qual_type)->getNamedType());
            return elaborated_clang_type.GetChildCompilerTypeAtIndex (exe_ctx,
                                                                      idx,
                                                                      transparent_pointers,
                                                                      omit_empty_base_classes,
                                                                      ignore_array_bounds,
                                                                      child_name,
                                                                      child_byte_size,
                                                                      child_byte_offset,
                                                                      child_bitfield_bit_size,
                                                                      child_bitfield_bit_offset,
                                                                      child_is_base_class,
                                                                      child_is_deref_of_parent,
                                                                      valobj,
                                                                      language_flags);
        }
            
        case clang::Type::Paren:
        {
            CompilerType paren_clang_type (getASTContext(), llvm::cast<clang::ParenType>(parent_qual_type)->desugar());
            return paren_clang_type.GetChildCompilerTypeAtIndex (exe_ctx,
                                                                 idx,
                                                                 transparent_pointers,
                                                                 omit_empty_base_classes,
                                                                 ignore_array_bounds,
                                                                 child_name,
                                                                 child_byte_size,
                                                                 child_byte_offset,
                                                                 child_bitfield_bit_size,
                                                                 child_bitfield_bit_offset,
                                                                 child_is_base_class,
                                                                 child_is_deref_of_parent,
                                                                 valobj,
                                                                 language_flags);
        }
            
            
        default:
            break;
    }
    return CompilerType();
}

static uint32_t
GetIndexForRecordBase
(
 const clang::RecordDecl *record_decl,
 const clang::CXXBaseSpecifier *base_spec,
 bool omit_empty_base_classes
 )
{
    uint32_t child_idx = 0;
    
    const clang::CXXRecordDecl *cxx_record_decl = llvm::dyn_cast<clang::CXXRecordDecl>(record_decl);
    
    //    const char *super_name = record_decl->getNameAsCString();
    //    const char *base_name = base_spec->getType()->getAs<clang::RecordType>()->getDecl()->getNameAsCString();
    //    printf ("GetIndexForRecordChild (%s, %s)\n", super_name, base_name);
    //
    if (cxx_record_decl)
    {
        clang::CXXRecordDecl::base_class_const_iterator base_class, base_class_end;
        for (base_class = cxx_record_decl->bases_begin(), base_class_end = cxx_record_decl->bases_end();
             base_class != base_class_end;
             ++base_class)
        {
            if (omit_empty_base_classes)
            {
                if (BaseSpecifierIsEmpty (base_class))
                    continue;
            }
            
            //            printf ("GetIndexForRecordChild (%s, %s) base[%u] = %s\n", super_name, base_name,
            //                    child_idx,
            //                    base_class->getType()->getAs<clang::RecordType>()->getDecl()->getNameAsCString());
            //
            //
            if (base_class == base_spec)
                return child_idx;
            ++child_idx;
        }
    }
    
    return UINT32_MAX;
}


static uint32_t
GetIndexForRecordChild (const clang::RecordDecl *record_decl,
                        clang::NamedDecl *canonical_decl,
                        bool omit_empty_base_classes)
{
    uint32_t child_idx = ClangASTContext::GetNumBaseClasses (llvm::dyn_cast<clang::CXXRecordDecl>(record_decl),
                                                             omit_empty_base_classes);
    
    clang::RecordDecl::field_iterator field, field_end;
    for (field = record_decl->field_begin(), field_end = record_decl->field_end();
         field != field_end;
         ++field, ++child_idx)
    {
        if (field->getCanonicalDecl() == canonical_decl)
            return child_idx;
    }
    
    return UINT32_MAX;
}

// Look for a child member (doesn't include base classes, but it does include
// their members) in the type hierarchy. Returns an index path into "clang_type"
// on how to reach the appropriate member.
//
//    class A
//    {
//    public:
//        int m_a;
//        int m_b;
//    };
//
//    class B
//    {
//    };
//
//    class C :
//        public B,
//        public A
//    {
//    };
//
// If we have a clang type that describes "class C", and we wanted to looked
// "m_b" in it:
//
// With omit_empty_base_classes == false we would get an integer array back with:
// { 1,  1 }
// The first index 1 is the child index for "class A" within class C
// The second index 1 is the child index for "m_b" within class A
//
// With omit_empty_base_classes == true we would get an integer array back with:
// { 0,  1 }
// The first index 0 is the child index for "class A" within class C (since class B doesn't have any members it doesn't count)
// The second index 1 is the child index for "m_b" within class A

size_t
ClangASTContext::GetIndexOfChildMemberWithName (lldb::opaque_compiler_type_t type, const char *name,
                                                   bool omit_empty_base_classes,
                                                   std::vector<uint32_t>& child_indexes)
{
    if (type && name && name[0])
    {
        clang::QualType qual_type(GetCanonicalQualType(type));
        const clang::Type::TypeClass type_class = qual_type->getTypeClass();
        switch (type_class)
        {
            case clang::Type::Record:
                if (GetCompleteType(type))
                {
                    const clang::RecordType *record_type = llvm::cast<clang::RecordType>(qual_type.getTypePtr());
                    const clang::RecordDecl *record_decl = record_type->getDecl();
                    
                    assert(record_decl);
                    uint32_t child_idx = 0;
                    
                    const clang::CXXRecordDecl *cxx_record_decl = llvm::dyn_cast<clang::CXXRecordDecl>(record_decl);
                    
                    // Try and find a field that matches NAME
                    clang::RecordDecl::field_iterator field, field_end;
                    llvm::StringRef name_sref(name);
                    for (field = record_decl->field_begin(), field_end = record_decl->field_end();
                         field != field_end;
                         ++field, ++child_idx)
                    {
                        llvm::StringRef field_name = field->getName();
                        if (field_name.empty())
                        {
                            CompilerType field_type(getASTContext(),field->getType());
                            child_indexes.push_back(child_idx);
                            if (field_type.GetIndexOfChildMemberWithName(name,  omit_empty_base_classes, child_indexes))
                                return child_indexes.size();
                            child_indexes.pop_back();
                            
                        }
                        else if (field_name.equals (name_sref))
                        {
                            // We have to add on the number of base classes to this index!
                            child_indexes.push_back (child_idx + ClangASTContext::GetNumBaseClasses (cxx_record_decl, omit_empty_base_classes));
                            return child_indexes.size();
                        }
                    }
                    
                    if (cxx_record_decl)
                    {
                        const clang::RecordDecl *parent_record_decl = cxx_record_decl;
                        
                        //printf ("parent = %s\n", parent_record_decl->getNameAsCString());
                        
                        //const Decl *root_cdecl = cxx_record_decl->getCanonicalDecl();
                        // Didn't find things easily, lets let clang do its thang...
                        clang::IdentifierInfo & ident_ref = getASTContext()->Idents.get(name_sref);
                        clang::DeclarationName decl_name(&ident_ref);
                        
                        clang::CXXBasePaths paths;
                        if (cxx_record_decl->lookupInBases([decl_name](const clang::CXXBaseSpecifier *specifier, clang::CXXBasePath &path) {
                                                               return clang::CXXRecordDecl::FindOrdinaryMember(specifier, path, decl_name);
                                                           },
                                                           paths))
                        {
                            clang::CXXBasePaths::const_paths_iterator path, path_end = paths.end();
                            for (path = paths.begin(); path != path_end; ++path)
                            {
                                const size_t num_path_elements = path->size();
                                for (size_t e=0; e<num_path_elements; ++e)
                                {
                                    clang::CXXBasePathElement elem = (*path)[e];
                                    
                                    child_idx = GetIndexForRecordBase (parent_record_decl, elem.Base, omit_empty_base_classes);
                                    if (child_idx == UINT32_MAX)
                                    {
                                        child_indexes.clear();
                                        return 0;
                                    }
                                    else
                                    {
                                        child_indexes.push_back (child_idx);
                                        parent_record_decl = llvm::cast<clang::RecordDecl>(elem.Base->getType()->getAs<clang::RecordType>()->getDecl());
                                    }
                                }
                                for (clang::NamedDecl *path_decl : path->Decls)
                                {
                                    child_idx = GetIndexForRecordChild (parent_record_decl, path_decl, omit_empty_base_classes);
                                    if (child_idx == UINT32_MAX)
                                    {
                                        child_indexes.clear();
                                        return 0;
                                    }
                                    else
                                    {
                                        child_indexes.push_back (child_idx);
                                    }
                                }
                            }
                            return child_indexes.size();
                        }
                    }
                    
                }
                break;
                
            case clang::Type::ObjCObject:
            case clang::Type::ObjCInterface:
                if (GetCompleteType(type))
                {
                    llvm::StringRef name_sref(name);
                    const clang::ObjCObjectType *objc_class_type = llvm::dyn_cast<clang::ObjCObjectType>(qual_type.getTypePtr());
                    assert (objc_class_type);
                    if (objc_class_type)
                    {
                        uint32_t child_idx = 0;
                        clang::ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterface();
                        
                        if (class_interface_decl)
                        {
                            clang::ObjCInterfaceDecl::ivar_iterator ivar_pos, ivar_end = class_interface_decl->ivar_end();
                            clang::ObjCInterfaceDecl *superclass_interface_decl = class_interface_decl->getSuperClass();
                            
                            for (ivar_pos = class_interface_decl->ivar_begin(); ivar_pos != ivar_end; ++ivar_pos, ++child_idx)
                            {
                                const clang::ObjCIvarDecl* ivar_decl = *ivar_pos;
                                
                                if (ivar_decl->getName().equals (name_sref))
                                {
                                    if ((!omit_empty_base_classes && superclass_interface_decl) ||
                                        ( omit_empty_base_classes && ObjCDeclHasIVars (superclass_interface_decl, true)))
                                        ++child_idx;
                                    
                                    child_indexes.push_back (child_idx);
                                    return child_indexes.size();
                                }
                            }
                            
                            if (superclass_interface_decl)
                            {
                                // The super class index is always zero for ObjC classes,
                                // so we push it onto the child indexes in case we find
                                // an ivar in our superclass...
                                child_indexes.push_back (0);
                                
                                CompilerType superclass_clang_type (getASTContext(), getASTContext()->getObjCInterfaceType(superclass_interface_decl));
                                if (superclass_clang_type.GetIndexOfChildMemberWithName (name,
                                                                                         omit_empty_base_classes,
                                                                                         child_indexes))
                                {
                                    // We did find an ivar in a superclass so just
                                    // return the results!
                                    return child_indexes.size();
                                }
                                
                                // We didn't find an ivar matching "name" in our
                                // superclass, pop the superclass zero index that
                                // we pushed on above.
                                child_indexes.pop_back();
                            }
                        }
                    }
                }
                break;
                
            case clang::Type::ObjCObjectPointer:
            {
                CompilerType objc_object_clang_type (getASTContext(), llvm::cast<clang::ObjCObjectPointerType>(qual_type.getTypePtr())->getPointeeType());
                return objc_object_clang_type.GetIndexOfChildMemberWithName (name,
                                                                             omit_empty_base_classes,
                                                                             child_indexes);
            }
                break;
                
                
            case clang::Type::ConstantArray:
            {
                //                const clang::ConstantArrayType *array = llvm::cast<clang::ConstantArrayType>(parent_qual_type.getTypePtr());
                //                const uint64_t element_count = array->getSize().getLimitedValue();
                //
                //                if (idx < element_count)
                //                {
                //                    std::pair<uint64_t, unsigned> field_type_info = ast->getTypeInfo(array->getElementType());
                //
                //                    char element_name[32];
                //                    ::snprintf (element_name, sizeof (element_name), "%s[%u]", parent_name ? parent_name : "", idx);
                //
                //                    child_name.assign(element_name);
                //                    assert(field_type_info.first % 8 == 0);
                //                    child_byte_size = field_type_info.first / 8;
                //                    child_byte_offset = idx * child_byte_size;
                //                    return array->getElementType().getAsOpaquePtr();
                //                }
            }
                break;
                
                //        case clang::Type::MemberPointerType:
                //            {
                //                MemberPointerType *mem_ptr_type = llvm::cast<MemberPointerType>(qual_type.getTypePtr());
                //                clang::QualType pointee_type = mem_ptr_type->getPointeeType();
                //
                //                if (ClangASTContext::IsAggregateType (pointee_type.getAsOpaquePtr()))
                //                {
                //                    return GetIndexOfChildWithName (ast,
                //                                                    mem_ptr_type->getPointeeType().getAsOpaquePtr(),
                //                                                    name);
                //                }
                //            }
                //            break;
                //
            case clang::Type::LValueReference:
            case clang::Type::RValueReference:
            {
                const clang::ReferenceType *reference_type = llvm::cast<clang::ReferenceType>(qual_type.getTypePtr());
                clang::QualType pointee_type(reference_type->getPointeeType());
                CompilerType pointee_clang_type (getASTContext(), pointee_type);
                
                if (pointee_clang_type.IsAggregateType ())
                {
                    return pointee_clang_type.GetIndexOfChildMemberWithName (name,
                                                                             omit_empty_base_classes,
                                                                             child_indexes);
                }
            }
                break;
                
            case clang::Type::Pointer:
            {
                CompilerType pointee_clang_type (GetPointeeType(type));
                
                if (pointee_clang_type.IsAggregateType ())
                {
                    return pointee_clang_type.GetIndexOfChildMemberWithName (name,
                                                                             omit_empty_base_classes,
                                                                             child_indexes);
                }
            }
                break;
                
            case clang::Type::Typedef:
                return CompilerType (getASTContext(), llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType()).GetIndexOfChildMemberWithName (name,
                                                                                                                                                                omit_empty_base_classes,
                                                                                                                                                                child_indexes);

            case clang::Type::Auto:
                return CompilerType (getASTContext(), llvm::cast<clang::AutoType>(qual_type)->getDeducedType()).GetIndexOfChildMemberWithName (name,
                                                                                                                                               omit_empty_base_classes,
                                                                                                                                               child_indexes);
                
            case clang::Type::Elaborated:
                return CompilerType (getASTContext(), llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType()).GetIndexOfChildMemberWithName (name,
                                                                                                                                                   omit_empty_base_classes,
                                                                                                                                                   child_indexes);
                
            case clang::Type::Paren:
                return CompilerType (getASTContext(), llvm::cast<clang::ParenType>(qual_type)->desugar()).GetIndexOfChildMemberWithName (name,
                                                                                                                                         omit_empty_base_classes,
                                                                                                                                         child_indexes);
                
            default:
                break;
        }
    }
    return 0;
}


// Get the index of the child of "clang_type" whose name matches. This function
// doesn't descend into the children, but only looks one level deep and name
// matches can include base class names.

uint32_t
ClangASTContext::GetIndexOfChildWithName (lldb::opaque_compiler_type_t type, const char *name, bool omit_empty_base_classes)
{
    if (type && name && name[0])
    {
        clang::QualType qual_type(GetCanonicalQualType(type));
        
        const clang::Type::TypeClass type_class = qual_type->getTypeClass();
        
        switch (type_class)
        {
            case clang::Type::Record:
                if (GetCompleteType(type))
                {
                    const clang::RecordType *record_type = llvm::cast<clang::RecordType>(qual_type.getTypePtr());
                    const clang::RecordDecl *record_decl = record_type->getDecl();
                    
                    assert(record_decl);
                    uint32_t child_idx = 0;
                    
                    const clang::CXXRecordDecl *cxx_record_decl = llvm::dyn_cast<clang::CXXRecordDecl>(record_decl);
                    
                    if (cxx_record_decl)
                    {
                        clang::CXXRecordDecl::base_class_const_iterator base_class, base_class_end;
                        for (base_class = cxx_record_decl->bases_begin(), base_class_end = cxx_record_decl->bases_end();
                             base_class != base_class_end;
                             ++base_class)
                        {
                            // Skip empty base classes
                            clang::CXXRecordDecl *base_class_decl = llvm::cast<clang::CXXRecordDecl>(base_class->getType()->getAs<clang::RecordType>()->getDecl());
                            if (omit_empty_base_classes && ClangASTContext::RecordHasFields(base_class_decl) == false)
                                continue;
                            
                            CompilerType base_class_clang_type (getASTContext(), base_class->getType());
                            std::string base_class_type_name (base_class_clang_type.GetTypeName().AsCString(""));
                            if (base_class_type_name.compare (name) == 0)
                                return child_idx;
                            ++child_idx;
                        }
                    }
                    
                    // Try and find a field that matches NAME
                    clang::RecordDecl::field_iterator field, field_end;
                    llvm::StringRef name_sref(name);
                    for (field = record_decl->field_begin(), field_end = record_decl->field_end();
                         field != field_end;
                         ++field, ++child_idx)
                    {
                        if (field->getName().equals (name_sref))
                            return child_idx;
                    }
                    
                }
                break;
                
            case clang::Type::ObjCObject:
            case clang::Type::ObjCInterface:
                if (GetCompleteType(type))
                {
                    llvm::StringRef name_sref(name);
                    const clang::ObjCObjectType *objc_class_type = llvm::dyn_cast<clang::ObjCObjectType>(qual_type.getTypePtr());
                    assert (objc_class_type);
                    if (objc_class_type)
                    {
                        uint32_t child_idx = 0;
                        clang::ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterface();
                        
                        if (class_interface_decl)
                        {
                            clang::ObjCInterfaceDecl::ivar_iterator ivar_pos, ivar_end = class_interface_decl->ivar_end();
                            clang::ObjCInterfaceDecl *superclass_interface_decl = class_interface_decl->getSuperClass();
                            
                            for (ivar_pos = class_interface_decl->ivar_begin(); ivar_pos != ivar_end; ++ivar_pos, ++child_idx)
                            {
                                const clang::ObjCIvarDecl* ivar_decl = *ivar_pos;
                                
                                if (ivar_decl->getName().equals (name_sref))
                                {
                                    if ((!omit_empty_base_classes && superclass_interface_decl) ||
                                        ( omit_empty_base_classes && ObjCDeclHasIVars (superclass_interface_decl, true)))
                                        ++child_idx;
                                    
                                    return child_idx;
                                }
                            }
                            
                            if (superclass_interface_decl)
                            {
                                if (superclass_interface_decl->getName().equals (name_sref))
                                    return 0;
                            }
                        }
                    }
                }
                break;
                
            case clang::Type::ObjCObjectPointer:
            {
                CompilerType pointee_clang_type (getASTContext(), llvm::cast<clang::ObjCObjectPointerType>(qual_type.getTypePtr())->getPointeeType());
                return pointee_clang_type.GetIndexOfChildWithName (name, omit_empty_base_classes);
            }
                break;
                
            case clang::Type::ConstantArray:
            {
                //                const clang::ConstantArrayType *array = llvm::cast<clang::ConstantArrayType>(parent_qual_type.getTypePtr());
                //                const uint64_t element_count = array->getSize().getLimitedValue();
                //
                //                if (idx < element_count)
                //                {
                //                    std::pair<uint64_t, unsigned> field_type_info = ast->getTypeInfo(array->getElementType());
                //
                //                    char element_name[32];
                //                    ::snprintf (element_name, sizeof (element_name), "%s[%u]", parent_name ? parent_name : "", idx);
                //
                //                    child_name.assign(element_name);
                //                    assert(field_type_info.first % 8 == 0);
                //                    child_byte_size = field_type_info.first / 8;
                //                    child_byte_offset = idx * child_byte_size;
                //                    return array->getElementType().getAsOpaquePtr();
                //                }
            }
                break;
                
                //        case clang::Type::MemberPointerType:
                //            {
                //                MemberPointerType *mem_ptr_type = llvm::cast<MemberPointerType>(qual_type.getTypePtr());
                //                clang::QualType pointee_type = mem_ptr_type->getPointeeType();
                //
                //                if (ClangASTContext::IsAggregateType (pointee_type.getAsOpaquePtr()))
                //                {
                //                    return GetIndexOfChildWithName (ast,
                //                                                    mem_ptr_type->getPointeeType().getAsOpaquePtr(),
                //                                                    name);
                //                }
                //            }
                //            break;
                //
            case clang::Type::LValueReference:
            case clang::Type::RValueReference:
            {
                const clang::ReferenceType *reference_type = llvm::cast<clang::ReferenceType>(qual_type.getTypePtr());
                CompilerType pointee_type (getASTContext(), reference_type->getPointeeType());
                
                if (pointee_type.IsAggregateType ())
                {
                    return pointee_type.GetIndexOfChildWithName (name, omit_empty_base_classes);
                }
            }
                break;
                
            case clang::Type::Pointer:
            {
                const clang::PointerType *pointer_type = llvm::cast<clang::PointerType>(qual_type.getTypePtr());
                CompilerType pointee_type (getASTContext(), pointer_type->getPointeeType());
                
                if (pointee_type.IsAggregateType ())
                {
                    return pointee_type.GetIndexOfChildWithName (name, omit_empty_base_classes);
                }
                else
                {
                    //                    if (parent_name)
                    //                    {
                    //                        child_name.assign(1, '*');
                    //                        child_name += parent_name;
                    //                    }
                    //
                    //                    // We have a pointer to an simple type
                    //                    if (idx == 0)
                    //                    {
                    //                        std::pair<uint64_t, unsigned> clang_type_info = ast->getTypeInfo(pointee_type);
                    //                        assert(clang_type_info.first % 8 == 0);
                    //                        child_byte_size = clang_type_info.first / 8;
                    //                        child_byte_offset = 0;
                    //                        return pointee_type.getAsOpaquePtr();
                    //                    }
                }
            }
                break;

            case clang::Type::Auto:
                return CompilerType (getASTContext(), llvm::cast<clang::AutoType>(qual_type)->getDeducedType()).GetIndexOfChildWithName (name, omit_empty_base_classes);
                
            case clang::Type::Elaborated:
                return CompilerType (getASTContext(), llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType()).GetIndexOfChildWithName (name, omit_empty_base_classes);
                
            case clang::Type::Paren:
                return CompilerType (getASTContext(), llvm::cast<clang::ParenType>(qual_type)->desugar()).GetIndexOfChildWithName (name, omit_empty_base_classes);
                
            case clang::Type::Typedef:
                return CompilerType (getASTContext(), llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType()).GetIndexOfChildWithName (name, omit_empty_base_classes);
                
            default:
                break;
        }
    }
    return UINT32_MAX;
}


size_t
ClangASTContext::GetNumTemplateArguments (lldb::opaque_compiler_type_t type)
{
    if (!type)
        return 0;

    clang::QualType qual_type (GetCanonicalQualType(type));
    const clang::Type::TypeClass type_class = qual_type->getTypeClass();
    switch (type_class)
    {
        case clang::Type::Record:
            if (GetCompleteType(type))
            {
                const clang::CXXRecordDecl *cxx_record_decl = qual_type->getAsCXXRecordDecl();
                if (cxx_record_decl)
                {
                    const clang::ClassTemplateSpecializationDecl *template_decl = llvm::dyn_cast<clang::ClassTemplateSpecializationDecl>(cxx_record_decl);
                    if (template_decl)
                        return template_decl->getTemplateArgs().size();
                }
            }
            break;
            
        case clang::Type::Typedef:
            return (CompilerType (getASTContext(), llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType())).GetNumTemplateArguments();

        case clang::Type::Auto:
            return (CompilerType (getASTContext(), llvm::cast<clang::AutoType>(qual_type)->getDeducedType())).GetNumTemplateArguments();
            
        case clang::Type::Elaborated:
            return (CompilerType (getASTContext(), llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType())).GetNumTemplateArguments();
            
        case clang::Type::Paren:
            return (CompilerType (getASTContext(), llvm::cast<clang::ParenType>(qual_type)->desugar())).GetNumTemplateArguments();
            
        default:
            break;
    }

    return 0;
}

CompilerType
ClangASTContext::GetTemplateArgument (lldb::opaque_compiler_type_t type, size_t arg_idx, lldb::TemplateArgumentKind &kind)
{
    if (!type)
        return CompilerType();

    clang::QualType qual_type (GetCanonicalQualType(type));
    const clang::Type::TypeClass type_class = qual_type->getTypeClass();
    switch (type_class)
    {
        case clang::Type::Record:
            if (GetCompleteType(type))
            {
                const clang::CXXRecordDecl *cxx_record_decl = qual_type->getAsCXXRecordDecl();
                if (cxx_record_decl)
                {
                    const clang::ClassTemplateSpecializationDecl *template_decl = llvm::dyn_cast<clang::ClassTemplateSpecializationDecl>(cxx_record_decl);
                    if (template_decl && arg_idx < template_decl->getTemplateArgs().size())
                    {
                        const clang::TemplateArgument &template_arg = template_decl->getTemplateArgs()[arg_idx];
                        switch (template_arg.getKind())
                        {
                            case clang::TemplateArgument::Null:
                                kind = eTemplateArgumentKindNull;
                                return CompilerType();
                                
                            case clang::TemplateArgument::Type:
                                kind = eTemplateArgumentKindType;
                                return CompilerType(getASTContext(), template_arg.getAsType());
                                
                            case clang::TemplateArgument::Declaration:
                                kind = eTemplateArgumentKindDeclaration;
                                return CompilerType();
                                
                            case clang::TemplateArgument::Integral:
                                kind = eTemplateArgumentKindIntegral;
                                return CompilerType(getASTContext(), template_arg.getIntegralType());
                                
                            case clang::TemplateArgument::Template:
                                kind = eTemplateArgumentKindTemplate;
                                return CompilerType();
                                
                            case clang::TemplateArgument::TemplateExpansion:
                                kind = eTemplateArgumentKindTemplateExpansion;
                                return CompilerType();
                                
                            case clang::TemplateArgument::Expression:
                                kind = eTemplateArgumentKindExpression;
                                return CompilerType();
                                
                            case clang::TemplateArgument::Pack:
                                kind = eTemplateArgumentKindPack;
                                return CompilerType();
                                
                            default:
                                assert (!"Unhandled clang::TemplateArgument::ArgKind");
                                break;
                        }
                    }
                }
            }
            break;
            
        case clang::Type::Typedef:
            return (CompilerType (getASTContext(), llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType())).GetTemplateArgument(arg_idx, kind);

        case clang::Type::Auto:
            return (CompilerType (getASTContext(), llvm::cast<clang::AutoType>(qual_type)->getDeducedType())).GetTemplateArgument(arg_idx, kind);
            
        case clang::Type::Elaborated:
            return (CompilerType (getASTContext(), llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType())).GetTemplateArgument(arg_idx, kind);
            
        case clang::Type::Paren:
            return (CompilerType (getASTContext(), llvm::cast<clang::ParenType>(qual_type)->desugar())).GetTemplateArgument(arg_idx, kind);
            
        default:
            break;
    }
    kind = eTemplateArgumentKindNull;
    return CompilerType ();
}

CompilerType
ClangASTContext::GetTypeForFormatters (void* type)
{
    if (type)
        return RemoveFastQualifiers(CompilerType(this, type));
    return CompilerType();
}

static bool
IsOperator (const char *name, clang::OverloadedOperatorKind &op_kind)
{
    if (name == nullptr || name[0] == '\0')
        return false;
    
#define OPERATOR_PREFIX "operator"
#define OPERATOR_PREFIX_LENGTH (sizeof (OPERATOR_PREFIX) - 1)
    
    const char *post_op_name = nullptr;
    
    bool no_space = true;
    
    if (::strncmp(name, OPERATOR_PREFIX, OPERATOR_PREFIX_LENGTH))
        return false;
    
    post_op_name = name + OPERATOR_PREFIX_LENGTH;
    
    if (post_op_name[0] == ' ')
    {
        post_op_name++;
        no_space = false;
    }
    
#undef OPERATOR_PREFIX
#undef OPERATOR_PREFIX_LENGTH
    
    // This is an operator, set the overloaded operator kind to invalid
    // in case this is a conversion operator...
    op_kind = clang::NUM_OVERLOADED_OPERATORS;
    
    switch (post_op_name[0])
    {
        default:
            if (no_space)
                return false;
            break;
        case 'n':
            if (no_space)
                return false;
            if  (strcmp (post_op_name, "new") == 0)
                op_kind = clang::OO_New;
            else if (strcmp (post_op_name, "new[]") == 0)
                op_kind = clang::OO_Array_New;
            break;
            
        case 'd':
            if (no_space)
                return false;
            if (strcmp (post_op_name, "delete") == 0)
                op_kind = clang::OO_Delete;
            else if (strcmp (post_op_name, "delete[]") == 0)
                op_kind = clang::OO_Array_Delete;
            break;
            
        case '+':
            if (post_op_name[1] == '\0')
                op_kind = clang::OO_Plus;
            else if (post_op_name[2] == '\0')
            {
                if (post_op_name[1] == '=')
                    op_kind = clang::OO_PlusEqual;
                else if (post_op_name[1] == '+')
                    op_kind = clang::OO_PlusPlus;
            }
            break;
            
        case '-':
            if (post_op_name[1] == '\0')
                op_kind = clang::OO_Minus;
            else if (post_op_name[2] == '\0')
            {
                switch (post_op_name[1])
                {
                    case '=': op_kind = clang::OO_MinusEqual; break;
                    case '-': op_kind = clang::OO_MinusMinus; break;
                    case '>': op_kind = clang::OO_Arrow; break;
                }
            }
            else if (post_op_name[3] == '\0')
            {
                if (post_op_name[2] == '*')
                    op_kind = clang::OO_ArrowStar; break;
            }
            break;
            
        case '*':
            if (post_op_name[1] == '\0')
                op_kind = clang::OO_Star;
            else if (post_op_name[1] == '=' && post_op_name[2] == '\0')
                op_kind = clang::OO_StarEqual;
            break;
            
        case '/':
            if (post_op_name[1] == '\0')
                op_kind = clang::OO_Slash;
            else if (post_op_name[1] == '=' && post_op_name[2] == '\0')
                op_kind = clang::OO_SlashEqual;
            break;
            
        case '%':
            if (post_op_name[1] == '\0')
                op_kind = clang::OO_Percent;
            else if (post_op_name[1] == '=' && post_op_name[2] == '\0')
                op_kind = clang::OO_PercentEqual;
            break;
            
            
        case '^':
            if (post_op_name[1] == '\0')
                op_kind = clang::OO_Caret;
            else if (post_op_name[1] == '=' && post_op_name[2] == '\0')
                op_kind = clang::OO_CaretEqual;
            break;
            
        case '&':
            if (post_op_name[1] == '\0')
                op_kind = clang::OO_Amp;
            else if (post_op_name[2] == '\0')
            {
                switch (post_op_name[1])
                {
                    case '=': op_kind = clang::OO_AmpEqual; break;
                    case '&': op_kind = clang::OO_AmpAmp; break;
                }
            }
            break;
            
        case '|':
            if (post_op_name[1] == '\0')
                op_kind = clang::OO_Pipe;
            else if (post_op_name[2] == '\0')
            {
                switch (post_op_name[1])
                {
                    case '=': op_kind = clang::OO_PipeEqual; break;
                    case '|': op_kind = clang::OO_PipePipe; break;
                }
            }
            break;
            
        case '~':
            if (post_op_name[1] == '\0')
                op_kind = clang::OO_Tilde;
            break;
            
        case '!':
            if (post_op_name[1] == '\0')
                op_kind = clang::OO_Exclaim;
            else if (post_op_name[1] == '=' && post_op_name[2] == '\0')
                op_kind = clang::OO_ExclaimEqual;
            break;
            
        case '=':
            if (post_op_name[1] == '\0')
                op_kind = clang::OO_Equal;
            else if (post_op_name[1] == '=' && post_op_name[2] == '\0')
                op_kind = clang::OO_EqualEqual;
            break;
            
        case '<':
            if (post_op_name[1] == '\0')
                op_kind = clang::OO_Less;
            else if (post_op_name[2] == '\0')
            {
                switch (post_op_name[1])
                {
                    case '<': op_kind = clang::OO_LessLess; break;
                    case '=': op_kind = clang::OO_LessEqual; break;
                }
            }
            else if (post_op_name[3] == '\0')
            {
                if (post_op_name[2] == '=')
                    op_kind = clang::OO_LessLessEqual;
            }
            break;
            
        case '>':
            if (post_op_name[1] == '\0')
                op_kind = clang::OO_Greater;
            else if (post_op_name[2] == '\0')
            {
                switch (post_op_name[1])
                {
                    case '>': op_kind = clang::OO_GreaterGreater; break;
                    case '=': op_kind = clang::OO_GreaterEqual; break;
                }
            }
            else if (post_op_name[1] == '>' &&
                     post_op_name[2] == '=' &&
                     post_op_name[3] == '\0')
            {
                op_kind = clang::OO_GreaterGreaterEqual;
            }
            break;
            
        case ',':
            if (post_op_name[1] == '\0')
                op_kind = clang::OO_Comma;
            break;
            
        case '(':
            if (post_op_name[1] == ')' && post_op_name[2] == '\0')
                op_kind = clang::OO_Call;
            break;
            
        case '[':
            if (post_op_name[1] == ']' && post_op_name[2] == '\0')
                op_kind = clang::OO_Subscript;
            break;
    }
    
    return true;
}

clang::EnumDecl *
ClangASTContext::GetAsEnumDecl (const CompilerType& type)
{
    const clang::EnumType *enutype = llvm::dyn_cast<clang::EnumType>(GetCanonicalQualType(type));
    if (enutype)
        return enutype->getDecl();
    return NULL;
}

clang::RecordDecl *
ClangASTContext::GetAsRecordDecl (const CompilerType& type)
{
    const clang::RecordType *record_type = llvm::dyn_cast<clang::RecordType>(GetCanonicalQualType(type));
    if (record_type)
        return record_type->getDecl();
    return nullptr;
}

clang::TagDecl *
ClangASTContext::GetAsTagDecl (const CompilerType& type)
{
    clang::QualType qual_type = GetCanonicalQualType(type);
    if (qual_type.isNull())
        return nullptr;
    else
        return qual_type->getAsTagDecl();
}

clang::CXXRecordDecl *
ClangASTContext::GetAsCXXRecordDecl (lldb::opaque_compiler_type_t type)
{
    return GetCanonicalQualType(type)->getAsCXXRecordDecl();
}

clang::ObjCInterfaceDecl *
ClangASTContext::GetAsObjCInterfaceDecl (const CompilerType& type)
{
    const clang::ObjCObjectType *objc_class_type = llvm::dyn_cast<clang::ObjCObjectType>(GetCanonicalQualType(type));
    if (objc_class_type)
        return objc_class_type->getInterface();
    return nullptr;
}

clang::FieldDecl *
ClangASTContext::AddFieldToRecordType (const CompilerType& type, const char *name,
                                       const CompilerType &field_clang_type,
                                       AccessType access,
                                       uint32_t bitfield_bit_size)
{
    if (!type.IsValid() || !field_clang_type.IsValid())
        return nullptr;
    ClangASTContext *ast = llvm::dyn_cast_or_null<ClangASTContext>(type.GetTypeSystem());
    if (!ast)
        return nullptr;
    clang::ASTContext* clang_ast = ast->getASTContext();
    
    clang::FieldDecl *field = nullptr;
    
    clang::Expr *bit_width = nullptr;
    if (bitfield_bit_size != 0)
    {
        llvm::APInt bitfield_bit_size_apint(clang_ast->getTypeSize(clang_ast->IntTy), bitfield_bit_size);
        bit_width = new (*clang_ast)clang::IntegerLiteral (*clang_ast, bitfield_bit_size_apint, clang_ast->IntTy, clang::SourceLocation());
    }
    
    clang::RecordDecl *record_decl = ast->GetAsRecordDecl (type);
    if (record_decl)
    {
        field = clang::FieldDecl::Create (*clang_ast,
                                          record_decl,
                                          clang::SourceLocation(),
                                          clang::SourceLocation(),
                                          name ? &clang_ast->Idents.get(name) : nullptr,  // Identifier
                                          GetQualType(field_clang_type),             // Field type
                                          nullptr,                                    // TInfo *
                                          bit_width,                                  // BitWidth
                                          false,                                      // Mutable
                                          clang::ICIS_NoInit);                        // HasInit
        
        if (!name)
        {
            // Determine whether this field corresponds to an anonymous
            // struct or union.
            if (const clang::TagType *TagT = field->getType()->getAs<clang::TagType>()) {
                if (clang::RecordDecl *Rec = llvm::dyn_cast<clang::RecordDecl>(TagT->getDecl()))
                    if (!Rec->getDeclName()) {
                        Rec->setAnonymousStructOrUnion(true);
                        field->setImplicit();
                        
                    }
            }
        }
        
        if (field)
        {
            field->setAccess (ClangASTContext::ConvertAccessTypeToAccessSpecifier (access));
            
            record_decl->addDecl(field);
            
#ifdef LLDB_CONFIGURATION_DEBUG
            VerifyDecl(field);
#endif
        }
    }
    else
    {
        clang::ObjCInterfaceDecl *class_interface_decl = ast->GetAsObjCInterfaceDecl (type);
        
        if (class_interface_decl)
        {
            const bool is_synthesized = false;
            
            field_clang_type.GetCompleteType();
            
            field = clang::ObjCIvarDecl::Create (*clang_ast,
                                                 class_interface_decl,
                                                 clang::SourceLocation(),
                                                 clang::SourceLocation(),
                                                 name ? &clang_ast->Idents.get(name) : nullptr,   // Identifier
                                                 GetQualType(field_clang_type),           // Field type
                                                 nullptr,                                     // TypeSourceInfo *
                                                 ConvertAccessTypeToObjCIvarAccessControl (access),
                                                 bit_width,
                                                 is_synthesized);
            
            if (field)
            {
                class_interface_decl->addDecl(field);
                
#ifdef LLDB_CONFIGURATION_DEBUG
                VerifyDecl(field);
#endif
            }
        }
    }
    return field;
}

void
ClangASTContext::BuildIndirectFields (const CompilerType& type)
{
    if (!type)
        return;

    ClangASTContext *ast = llvm::dyn_cast<ClangASTContext>(type.GetTypeSystem());
    if (!ast)
        return;

    clang::RecordDecl *record_decl = ast->GetAsRecordDecl(type);
    
    if (!record_decl)
        return;
    
    typedef llvm::SmallVector <clang::IndirectFieldDecl *, 1> IndirectFieldVector;
    
    IndirectFieldVector indirect_fields;
    clang::RecordDecl::field_iterator field_pos;
    clang::RecordDecl::field_iterator field_end_pos = record_decl->field_end();
    clang::RecordDecl::field_iterator last_field_pos = field_end_pos;
    for (field_pos = record_decl->field_begin(); field_pos != field_end_pos; last_field_pos = field_pos++)
    {
        if (field_pos->isAnonymousStructOrUnion())
        {
            clang::QualType field_qual_type = field_pos->getType();
            
            const clang::RecordType *field_record_type = field_qual_type->getAs<clang::RecordType>();
            
            if (!field_record_type)
                continue;
            
            clang::RecordDecl *field_record_decl = field_record_type->getDecl();
            
            if (!field_record_decl)
                continue;
            
            for (clang::RecordDecl::decl_iterator di = field_record_decl->decls_begin(), de = field_record_decl->decls_end();
                 di != de;
                 ++di)
            {
                if (clang::FieldDecl *nested_field_decl = llvm::dyn_cast<clang::FieldDecl>(*di))
                {
                    clang::NamedDecl **chain = new (*ast->getASTContext()) clang::NamedDecl*[2];
                    chain[0] = *field_pos;
                    chain[1] = nested_field_decl;
                    clang::IndirectFieldDecl *indirect_field = clang::IndirectFieldDecl::Create(*ast->getASTContext(),
                                                                                                record_decl,
                                                                                                clang::SourceLocation(),
                                                                                                nested_field_decl->getIdentifier(),
                                                                                                nested_field_decl->getType(),
                                                                                                chain,
                                                                                                2);
                    
                    indirect_field->setImplicit();
                    
                    indirect_field->setAccess(ClangASTContext::UnifyAccessSpecifiers(field_pos->getAccess(),
                                                                                     nested_field_decl->getAccess()));
                    
                    indirect_fields.push_back(indirect_field);
                }
                else if (clang::IndirectFieldDecl *nested_indirect_field_decl = llvm::dyn_cast<clang::IndirectFieldDecl>(*di))
                {
                    int nested_chain_size = nested_indirect_field_decl->getChainingSize();
                    clang::NamedDecl **chain = new (*ast->getASTContext()) clang::NamedDecl*[nested_chain_size + 1];
                    chain[0] = *field_pos;
                    
                    int chain_index = 1;
                    for (clang::IndirectFieldDecl::chain_iterator nci = nested_indirect_field_decl->chain_begin(),
                         nce = nested_indirect_field_decl->chain_end();
                         nci < nce;
                         ++nci)
                    {
                        chain[chain_index] = *nci;
                        chain_index++;
                    }
                    
                    clang::IndirectFieldDecl *indirect_field = clang::IndirectFieldDecl::Create(*ast->getASTContext(),
                                                                                                record_decl,
                                                                                                clang::SourceLocation(),
                                                                                                nested_indirect_field_decl->getIdentifier(),
                                                                                                nested_indirect_field_decl->getType(),
                                                                                                chain,
                                                                                                nested_chain_size + 1);
                    
                    indirect_field->setImplicit();
                    
                    indirect_field->setAccess(ClangASTContext::UnifyAccessSpecifiers(field_pos->getAccess(),
                                                                                     nested_indirect_field_decl->getAccess()));
                    
                    indirect_fields.push_back(indirect_field);
                }
            }
        }
    }
    
    // Check the last field to see if it has an incomplete array type as its
    // last member and if it does, the tell the record decl about it
    if (last_field_pos != field_end_pos)
    {
        if (last_field_pos->getType()->isIncompleteArrayType())
            record_decl->hasFlexibleArrayMember();
    }
    
    for (IndirectFieldVector::iterator ifi = indirect_fields.begin(), ife = indirect_fields.end();
         ifi < ife;
         ++ifi)
    {
        record_decl->addDecl(*ifi);
    }
}

void
ClangASTContext::SetIsPacked (const CompilerType& type)
{
    if (type)
    {
        ClangASTContext *ast = llvm::dyn_cast<ClangASTContext>(type.GetTypeSystem());
        if (ast)
        {
            clang::RecordDecl *record_decl = GetAsRecordDecl(type);
    
            if (!record_decl)
                return;
    
            record_decl->addAttr(clang::PackedAttr::CreateImplicit(*ast->getASTContext()));
        }
    }
}

clang::VarDecl *
ClangASTContext::AddVariableToRecordType (const CompilerType& type, const char *name,
                                          const CompilerType &var_type,
                                          AccessType access)
{
    clang::VarDecl *var_decl = nullptr;
    
    if (!type.IsValid() || !var_type.IsValid())
        return nullptr;
    ClangASTContext *ast = llvm::dyn_cast<ClangASTContext>(type.GetTypeSystem());
    if (!ast)
        return nullptr;
    
    clang::RecordDecl *record_decl = ast->GetAsRecordDecl (type);
    if (record_decl)
    {
        var_decl = clang::VarDecl::Create (*ast->getASTContext(),                      // ASTContext &
                                           record_decl,                                // DeclContext *
                                           clang::SourceLocation(),                    // clang::SourceLocation StartLoc
                                           clang::SourceLocation(),                    // clang::SourceLocation IdLoc
                                           name ? &ast->getASTContext()->Idents.get(name) : nullptr,  // clang::IdentifierInfo *
                                           GetQualType(var_type),                      // Variable clang::QualType
                                           nullptr,                                    // TypeSourceInfo *
                                           clang::SC_Static);                          // StorageClass
        if (var_decl)
        {
            var_decl->setAccess(ClangASTContext::ConvertAccessTypeToAccessSpecifier (access));
            record_decl->addDecl(var_decl);
            
#ifdef LLDB_CONFIGURATION_DEBUG
            VerifyDecl(var_decl);
#endif
        }
    }
    return var_decl;
}


clang::CXXMethodDecl *
ClangASTContext::AddMethodToCXXRecordType (lldb::opaque_compiler_type_t type, const char *name,
                                              const CompilerType &method_clang_type,
                                              lldb::AccessType access,
                                              bool is_virtual,
                                              bool is_static,
                                              bool is_inline,
                                              bool is_explicit,
                                              bool is_attr_used,
                                              bool is_artificial)
{
    if (!type || !method_clang_type.IsValid() || name == nullptr || name[0] == '\0')
        return nullptr;
    
    clang::QualType record_qual_type(GetCanonicalQualType(type));
    
    clang::CXXRecordDecl *cxx_record_decl = record_qual_type->getAsCXXRecordDecl();
    
    if (cxx_record_decl == nullptr)
        return nullptr;
    
    clang::QualType method_qual_type (GetQualType(method_clang_type));
    
    clang::CXXMethodDecl *cxx_method_decl = nullptr;
    
    clang::DeclarationName decl_name (&getASTContext()->Idents.get(name));
    
    const clang::FunctionType *function_type = llvm::dyn_cast<clang::FunctionType>(method_qual_type.getTypePtr());
    
    if (function_type == nullptr)
        return nullptr;
    
    const clang::FunctionProtoType *method_function_prototype (llvm::dyn_cast<clang::FunctionProtoType>(function_type));
    
    if (!method_function_prototype)
        return nullptr;
    
    unsigned int num_params = method_function_prototype->getNumParams();
    
    clang::CXXDestructorDecl *cxx_dtor_decl(nullptr);
    clang::CXXConstructorDecl *cxx_ctor_decl(nullptr);
    
    if (is_artificial)
        return nullptr; // skip everything artificial
    
    if (name[0] == '~')
    {
        cxx_dtor_decl = clang::CXXDestructorDecl::Create (*getASTContext(),
                                                          cxx_record_decl,
                                                          clang::SourceLocation(),
                                                          clang::DeclarationNameInfo (getASTContext()->DeclarationNames.getCXXDestructorName (getASTContext()->getCanonicalType (record_qual_type)), clang::SourceLocation()),
                                                          method_qual_type,
                                                          nullptr,
                                                          is_inline,
                                                          is_artificial);
        cxx_method_decl = cxx_dtor_decl;
    }
    else if (decl_name == cxx_record_decl->getDeclName())
    {
        cxx_ctor_decl = clang::CXXConstructorDecl::Create (*getASTContext(),
                                                           cxx_record_decl,
                                                           clang::SourceLocation(),
                                                           clang::DeclarationNameInfo (getASTContext()->DeclarationNames.getCXXConstructorName (getASTContext()->getCanonicalType (record_qual_type)), clang::SourceLocation()),
                                                           method_qual_type,
                                                           nullptr, // TypeSourceInfo *
                                                           is_explicit,
                                                           is_inline,
                                                           is_artificial,
                                                           false /*is_constexpr*/);
        cxx_method_decl = cxx_ctor_decl;
    }
    else
    {
        clang::StorageClass SC = is_static ? clang::SC_Static : clang::SC_None;
        clang::OverloadedOperatorKind op_kind = clang::NUM_OVERLOADED_OPERATORS;
        
        if (IsOperator (name, op_kind))
        {
            if (op_kind != clang::NUM_OVERLOADED_OPERATORS)
            {
                // Check the number of operator parameters. Sometimes we have
                // seen bad DWARF that doesn't correctly describe operators and
                // if we try to create a method and add it to the class, clang
                // will assert and crash, so we need to make sure things are
                // acceptable.
                if (!ClangASTContext::CheckOverloadedOperatorKindParameterCount (op_kind, num_params))
                    return nullptr;
                cxx_method_decl = clang::CXXMethodDecl::Create (*getASTContext(),
                                                                cxx_record_decl,
                                                                clang::SourceLocation(),
                                                                clang::DeclarationNameInfo (getASTContext()->DeclarationNames.getCXXOperatorName (op_kind), clang::SourceLocation()),
                                                                method_qual_type,
                                                                nullptr, // TypeSourceInfo *
                                                                SC,
                                                                is_inline,
                                                                false /*is_constexpr*/,
                                                                clang::SourceLocation());
            }
            else if (num_params == 0)
            {
                // Conversion operators don't take params...
                cxx_method_decl = clang::CXXConversionDecl::Create (*getASTContext(),
                                                                    cxx_record_decl,
                                                                    clang::SourceLocation(),
                                                                    clang::DeclarationNameInfo (getASTContext()->DeclarationNames.getCXXConversionFunctionName (getASTContext()->getCanonicalType (function_type->getReturnType())), clang::SourceLocation()),
                                                                    method_qual_type,
                                                                    nullptr, // TypeSourceInfo *
                                                                    is_inline,
                                                                    is_explicit,
                                                                    false /*is_constexpr*/,
                                                                    clang::SourceLocation());
            }
        }
        
        if (cxx_method_decl == nullptr)
        {
            cxx_method_decl = clang::CXXMethodDecl::Create (*getASTContext(),
                                                            cxx_record_decl,
                                                            clang::SourceLocation(),
                                                            clang::DeclarationNameInfo (decl_name, clang::SourceLocation()),
                                                            method_qual_type,
                                                            nullptr, // TypeSourceInfo *
                                                            SC,
                                                            is_inline,
                                                            false /*is_constexpr*/,
                                                            clang::SourceLocation());
        }
    }
    
    clang::AccessSpecifier access_specifier = ClangASTContext::ConvertAccessTypeToAccessSpecifier (access);
    
    cxx_method_decl->setAccess (access_specifier);
    cxx_method_decl->setVirtualAsWritten (is_virtual);
    
    if (is_attr_used)
        cxx_method_decl->addAttr(clang::UsedAttr::CreateImplicit(*getASTContext()));
    
    // Populate the method decl with parameter decls
    
    llvm::SmallVector<clang::ParmVarDecl *, 12> params;
    
    for (unsigned param_index = 0;
         param_index < num_params;
         ++param_index)
    {
        params.push_back (clang::ParmVarDecl::Create (*getASTContext(),
                                                      cxx_method_decl,
                                                      clang::SourceLocation(),
                                                      clang::SourceLocation(),
                                                      nullptr, // anonymous
                                                      method_function_prototype->getParamType(param_index),
                                                      nullptr,
                                                      clang::SC_None,
                                                      nullptr));
    }
    
    cxx_method_decl->setParams (llvm::ArrayRef<clang::ParmVarDecl*>(params));
    
    cxx_record_decl->addDecl (cxx_method_decl);
    
    // Sometimes the debug info will mention a constructor (default/copy/move),
    // destructor, or assignment operator (copy/move) but there won't be any
    // version of this in the code. So we check if the function was artificially
    // generated and if it is trivial and this lets the compiler/backend know
    // that it can inline the IR for these when it needs to and we can avoid a
    // "missing function" error when running expressions.
    
    if (is_artificial)
    {
        if (cxx_ctor_decl &&
            ((cxx_ctor_decl->isDefaultConstructor() && cxx_record_decl->hasTrivialDefaultConstructor ()) ||
             (cxx_ctor_decl->isCopyConstructor()    && cxx_record_decl->hasTrivialCopyConstructor    ()) ||
             (cxx_ctor_decl->isMoveConstructor()    && cxx_record_decl->hasTrivialMoveConstructor    ()) ))
        {
            cxx_ctor_decl->setDefaulted();
            cxx_ctor_decl->setTrivial(true);
        }
        else if (cxx_dtor_decl)
        {
            if (cxx_record_decl->hasTrivialDestructor())
            {
                cxx_dtor_decl->setDefaulted();
                cxx_dtor_decl->setTrivial(true);
            }
        }
        else if ((cxx_method_decl->isCopyAssignmentOperator() && cxx_record_decl->hasTrivialCopyAssignment()) ||
                 (cxx_method_decl->isMoveAssignmentOperator() && cxx_record_decl->hasTrivialMoveAssignment()))
        {
            cxx_method_decl->setDefaulted();
            cxx_method_decl->setTrivial(true);
        }
    }
    
#ifdef LLDB_CONFIGURATION_DEBUG
    VerifyDecl(cxx_method_decl);
#endif
    
    //    printf ("decl->isPolymorphic()             = %i\n", cxx_record_decl->isPolymorphic());
    //    printf ("decl->isAggregate()               = %i\n", cxx_record_decl->isAggregate());
    //    printf ("decl->isPOD()                     = %i\n", cxx_record_decl->isPOD());
    //    printf ("decl->isEmpty()                   = %i\n", cxx_record_decl->isEmpty());
    //    printf ("decl->isAbstract()                = %i\n", cxx_record_decl->isAbstract());
    //    printf ("decl->hasTrivialConstructor()     = %i\n", cxx_record_decl->hasTrivialConstructor());
    //    printf ("decl->hasTrivialCopyConstructor() = %i\n", cxx_record_decl->hasTrivialCopyConstructor());
    //    printf ("decl->hasTrivialCopyAssignment()  = %i\n", cxx_record_decl->hasTrivialCopyAssignment());
    //    printf ("decl->hasTrivialDestructor()      = %i\n", cxx_record_decl->hasTrivialDestructor());
    return cxx_method_decl;
}


#pragma mark C++ Base Classes

clang::CXXBaseSpecifier *
ClangASTContext::CreateBaseClassSpecifier (lldb::opaque_compiler_type_t type, AccessType access, bool is_virtual, bool base_of_class)
{
    if (type)
        return new clang::CXXBaseSpecifier (clang::SourceRange(),
                                            is_virtual,
                                            base_of_class,
                                            ClangASTContext::ConvertAccessTypeToAccessSpecifier (access),
                                            getASTContext()->getTrivialTypeSourceInfo (GetQualType(type)),
                                            clang::SourceLocation());
    return nullptr;
}

void
ClangASTContext::DeleteBaseClassSpecifiers (clang::CXXBaseSpecifier **base_classes, unsigned num_base_classes)
{
    for (unsigned i=0; i<num_base_classes; ++i)
    {
        delete base_classes[i];
        base_classes[i] = nullptr;
    }
}

bool
ClangASTContext::SetBaseClassesForClassType (lldb::opaque_compiler_type_t type, clang::CXXBaseSpecifier const * const *base_classes,
                                                unsigned num_base_classes)
{
    if (type)
    {
        clang::CXXRecordDecl *cxx_record_decl = GetAsCXXRecordDecl(type);
        if (cxx_record_decl)
        {
            cxx_record_decl->setBases(base_classes, num_base_classes);
            return true;
        }
    }
    return false;
}

bool
ClangASTContext::SetObjCSuperClass (const CompilerType& type, const CompilerType &superclass_clang_type)
{
    ClangASTContext *ast = llvm::dyn_cast_or_null<ClangASTContext>(type.GetTypeSystem());
    if (!ast)
        return false;
    clang::ASTContext* clang_ast = ast->getASTContext();

    if (type && superclass_clang_type.IsValid() && superclass_clang_type.GetTypeSystem() == type.GetTypeSystem())
    {
        clang::ObjCInterfaceDecl *class_interface_decl = GetAsObjCInterfaceDecl (type);
        clang::ObjCInterfaceDecl *super_interface_decl = GetAsObjCInterfaceDecl (superclass_clang_type);
        if (class_interface_decl && super_interface_decl)
        {
            class_interface_decl->setSuperClass(clang_ast->getTrivialTypeSourceInfo(clang_ast->getObjCInterfaceType(super_interface_decl)));
            return true;
        }
    }
    return false;
}

bool
ClangASTContext::AddObjCClassProperty (const CompilerType& type,
                                       const char *property_name,
                                       const CompilerType &property_clang_type,
                                       clang::ObjCIvarDecl *ivar_decl,
                                       const char *property_setter_name,
                                       const char *property_getter_name,
                                       uint32_t property_attributes,
                                       ClangASTMetadata *metadata)
{
    if (!type || !property_clang_type.IsValid() || property_name == nullptr || property_name[0] == '\0')
        return false;
    ClangASTContext *ast = llvm::dyn_cast<ClangASTContext>(type.GetTypeSystem());
    if (!ast)
        return false;
    clang::ASTContext* clang_ast = ast->getASTContext();
    
    clang::ObjCInterfaceDecl *class_interface_decl = GetAsObjCInterfaceDecl (type);
    
    if (class_interface_decl)
    {
        CompilerType property_clang_type_to_access;
        
        if (property_clang_type.IsValid())
            property_clang_type_to_access = property_clang_type;
        else if (ivar_decl)
            property_clang_type_to_access = CompilerType (clang_ast, ivar_decl->getType());
        
        if (class_interface_decl && property_clang_type_to_access.IsValid())
        {
            clang::TypeSourceInfo *prop_type_source;
            if (ivar_decl)
                prop_type_source = clang_ast->getTrivialTypeSourceInfo (ivar_decl->getType());
            else
                prop_type_source = clang_ast->getTrivialTypeSourceInfo (GetQualType(property_clang_type));
            
            clang::ObjCPropertyDecl *property_decl = clang::ObjCPropertyDecl::Create (*clang_ast,
                                                                                      class_interface_decl,
                                                                                      clang::SourceLocation(), // Source Location
                                                                                      &clang_ast->Idents.get(property_name),
                                                                                      clang::SourceLocation(), //Source Location for AT
                                                                                      clang::SourceLocation(), //Source location for (
                                                                                      ivar_decl ? ivar_decl->getType() : ClangASTContext::GetQualType(property_clang_type),
                                                                                      prop_type_source);
            
            if (property_decl)
            {
                if (metadata)
                    ClangASTContext::SetMetadata(clang_ast, property_decl, *metadata);
                
                class_interface_decl->addDecl (property_decl);
                
                clang::Selector setter_sel, getter_sel;
                
                if (property_setter_name != nullptr)
                {
                    std::string property_setter_no_colon(property_setter_name, strlen(property_setter_name) - 1);
                    clang::IdentifierInfo *setter_ident = &clang_ast->Idents.get(property_setter_no_colon.c_str());
                    setter_sel = clang_ast->Selectors.getSelector(1, &setter_ident);
                }
                else if (!(property_attributes & DW_APPLE_PROPERTY_readonly))
                {
                    std::string setter_sel_string("set");
                    setter_sel_string.push_back(::toupper(property_name[0]));
                    setter_sel_string.append(&property_name[1]);
                    clang::IdentifierInfo *setter_ident = &clang_ast->Idents.get(setter_sel_string.c_str());
                    setter_sel = clang_ast->Selectors.getSelector(1, &setter_ident);
                }
                property_decl->setSetterName(setter_sel);
                property_decl->setPropertyAttributes (clang::ObjCPropertyDecl::OBJC_PR_setter);
                
                if (property_getter_name != nullptr)
                {
                    clang::IdentifierInfo *getter_ident = &clang_ast->Idents.get(property_getter_name);
                    getter_sel = clang_ast->Selectors.getSelector(0, &getter_ident);
                }
                else
                {
                    clang::IdentifierInfo *getter_ident = &clang_ast->Idents.get(property_name);
                    getter_sel = clang_ast->Selectors.getSelector(0, &getter_ident);
                }
                property_decl->setGetterName(getter_sel);
                property_decl->setPropertyAttributes (clang::ObjCPropertyDecl::OBJC_PR_getter);
                
                if (ivar_decl)
                    property_decl->setPropertyIvarDecl (ivar_decl);
                
                if (property_attributes & DW_APPLE_PROPERTY_readonly)
                    property_decl->setPropertyAttributes (clang::ObjCPropertyDecl::OBJC_PR_readonly);
                if (property_attributes & DW_APPLE_PROPERTY_readwrite)
                    property_decl->setPropertyAttributes (clang::ObjCPropertyDecl::OBJC_PR_readwrite);
                if (property_attributes & DW_APPLE_PROPERTY_assign)
                    property_decl->setPropertyAttributes (clang::ObjCPropertyDecl::OBJC_PR_assign);
                if (property_attributes & DW_APPLE_PROPERTY_retain)
                    property_decl->setPropertyAttributes (clang::ObjCPropertyDecl::OBJC_PR_retain);
                if (property_attributes & DW_APPLE_PROPERTY_copy)
                    property_decl->setPropertyAttributes (clang::ObjCPropertyDecl::OBJC_PR_copy);
                if (property_attributes & DW_APPLE_PROPERTY_nonatomic)
                    property_decl->setPropertyAttributes (clang::ObjCPropertyDecl::OBJC_PR_nonatomic);
                
                if (!getter_sel.isNull() && !class_interface_decl->lookupInstanceMethod(getter_sel))
                {
                    const bool isInstance = true;
                    const bool isVariadic = false;
                    const bool isSynthesized = false;
                    const bool isImplicitlyDeclared = true;
                    const bool isDefined = false;
                    const clang::ObjCMethodDecl::ImplementationControl impControl = clang::ObjCMethodDecl::None;
                    const bool HasRelatedResultType = false;
                    
                    clang::ObjCMethodDecl *getter = clang::ObjCMethodDecl::Create (*clang_ast,
                                                                                   clang::SourceLocation(),
                                                                                   clang::SourceLocation(),
                                                                                   getter_sel,
                                                                                   GetQualType(property_clang_type_to_access),
                                                                                   nullptr,
                                                                                   class_interface_decl,
                                                                                   isInstance,
                                                                                   isVariadic,
                                                                                   isSynthesized,
                                                                                   isImplicitlyDeclared,
                                                                                   isDefined,
                                                                                   impControl,
                                                                                   HasRelatedResultType);
                    
                    if (getter && metadata)
                        ClangASTContext::SetMetadata(clang_ast, getter, *metadata);
                    
                    if (getter)
                    {
                        getter->setMethodParams(*clang_ast, llvm::ArrayRef<clang::ParmVarDecl*>(), llvm::ArrayRef<clang::SourceLocation>());
                        
                        class_interface_decl->addDecl(getter);
                    }
                }
                
                if (!setter_sel.isNull() && !class_interface_decl->lookupInstanceMethod(setter_sel))
                {
                    clang::QualType result_type = clang_ast->VoidTy;
                    
                    const bool isInstance = true;
                    const bool isVariadic = false;
                    const bool isSynthesized = false;
                    const bool isImplicitlyDeclared = true;
                    const bool isDefined = false;
                    const clang::ObjCMethodDecl::ImplementationControl impControl = clang::ObjCMethodDecl::None;
                    const bool HasRelatedResultType = false;
                    
                    clang::ObjCMethodDecl *setter = clang::ObjCMethodDecl::Create (*clang_ast,
                                                                                   clang::SourceLocation(),
                                                                                   clang::SourceLocation(),
                                                                                   setter_sel,
                                                                                   result_type,
                                                                                   nullptr,
                                                                                   class_interface_decl,
                                                                                   isInstance,
                                                                                   isVariadic,
                                                                                   isSynthesized,
                                                                                   isImplicitlyDeclared,
                                                                                   isDefined,
                                                                                   impControl,
                                                                                   HasRelatedResultType);
                    
                    if (setter && metadata)
                        ClangASTContext::SetMetadata(clang_ast, setter, *metadata);
                    
                    llvm::SmallVector<clang::ParmVarDecl *, 1> params;
                    
                    params.push_back (clang::ParmVarDecl::Create (*clang_ast,
                                                                  setter,
                                                                  clang::SourceLocation(),
                                                                  clang::SourceLocation(),
                                                                  nullptr, // anonymous
                                                                  GetQualType(property_clang_type_to_access),
                                                                  nullptr,
                                                                  clang::SC_Auto,
                                                                  nullptr));
                    
                    if (setter)
                    {
                        setter->setMethodParams(*clang_ast, llvm::ArrayRef<clang::ParmVarDecl*>(params), llvm::ArrayRef<clang::SourceLocation>());
                        
                        class_interface_decl->addDecl(setter);
                    }
                }
                
                return true;
            }
        }
    }
    return false;
}

bool
ClangASTContext::IsObjCClassTypeAndHasIVars (const CompilerType& type, bool check_superclass)
{
    clang::ObjCInterfaceDecl *class_interface_decl = GetAsObjCInterfaceDecl (type);
    if (class_interface_decl)
        return ObjCDeclHasIVars (class_interface_decl, check_superclass);
    return false;
}


clang::ObjCMethodDecl *
ClangASTContext::AddMethodToObjCObjectType (const CompilerType& type,
                                            const char *name,  // the full symbol name as seen in the symbol table (lldb::opaque_compiler_type_t type, "-[NString stringWithCString:]")
                                            const CompilerType &method_clang_type,
                                            lldb::AccessType access,
                                            bool is_artificial)
{
    if (!type || !method_clang_type.IsValid())
        return nullptr;
    
    clang::ObjCInterfaceDecl *class_interface_decl = GetAsObjCInterfaceDecl(type);
    
    if (class_interface_decl == nullptr)
        return nullptr;
    ClangASTContext *lldb_ast = llvm::dyn_cast<ClangASTContext>(type.GetTypeSystem());
    if (lldb_ast == nullptr)
        return nullptr;
    clang::ASTContext *ast = lldb_ast->getASTContext();

    const char *selector_start = ::strchr (name, ' ');
    if (selector_start == nullptr)
        return nullptr;
    
    selector_start++;
    llvm::SmallVector<clang::IdentifierInfo *, 12> selector_idents;
    
    size_t len = 0;
    const char *start;
    //printf ("name = '%s'\n", name);
    
    unsigned num_selectors_with_args = 0;
    for (start = selector_start;
         start && *start != '\0' && *start != ']';
         start += len)
    {
        len = ::strcspn(start, ":]");
        bool has_arg = (start[len] == ':');
        if (has_arg)
            ++num_selectors_with_args;
        selector_idents.push_back (&ast->Idents.get (llvm::StringRef (start, len)));
        if (has_arg)
            len += 1;
    }
    
    
    if (selector_idents.size() == 0)
        return nullptr;
    
    clang::Selector method_selector = ast->Selectors.getSelector (num_selectors_with_args ? selector_idents.size() : 0,
                                                                    selector_idents.data());
    
    clang::QualType method_qual_type (GetQualType(method_clang_type));
    
    // Populate the method decl with parameter decls
    const clang::Type *method_type(method_qual_type.getTypePtr());
    
    if (method_type == nullptr)
        return nullptr;
    
    const clang::FunctionProtoType *method_function_prototype (llvm::dyn_cast<clang::FunctionProtoType>(method_type));
    
    if (!method_function_prototype)
        return nullptr;
    
    
    bool is_variadic = false;
    bool is_synthesized = false;
    bool is_defined = false;
    clang::ObjCMethodDecl::ImplementationControl imp_control = clang::ObjCMethodDecl::None;
    
    const unsigned num_args = method_function_prototype->getNumParams();
    
    if (num_args != num_selectors_with_args)
        return nullptr; // some debug information is corrupt.  We are not going to deal with it.
    
    clang::ObjCMethodDecl *objc_method_decl = clang::ObjCMethodDecl::Create (*ast,
                                                                             clang::SourceLocation(), // beginLoc,
                                                                             clang::SourceLocation(), // endLoc,
                                                                             method_selector,
                                                                             method_function_prototype->getReturnType(),
                                                                             nullptr, // TypeSourceInfo *ResultTInfo,
                                                                             ClangASTContext::GetASTContext(ast)->GetDeclContextForType(GetQualType(type)),
                                                                             name[0] == '-',
                                                                             is_variadic,
                                                                             is_synthesized,
                                                                             true, // is_implicitly_declared; we force this to true because we don't have source locations
                                                                             is_defined,
                                                                             imp_control,
                                                                             false /*has_related_result_type*/);
    
    
    if (objc_method_decl == nullptr)
        return nullptr;
    
    if (num_args > 0)
    {
        llvm::SmallVector<clang::ParmVarDecl *, 12> params;
        
        for (unsigned param_index = 0; param_index < num_args; ++param_index)
        {
            params.push_back (clang::ParmVarDecl::Create (*ast,
                                                          objc_method_decl,
                                                          clang::SourceLocation(),
                                                          clang::SourceLocation(),
                                                          nullptr, // anonymous
                                                          method_function_prototype->getParamType(param_index),
                                                          nullptr,
                                                          clang::SC_Auto,
                                                          nullptr));
        }
        
        objc_method_decl->setMethodParams(*ast, llvm::ArrayRef<clang::ParmVarDecl*>(params), llvm::ArrayRef<clang::SourceLocation>());
    }
    
    class_interface_decl->addDecl (objc_method_decl);
    
#ifdef LLDB_CONFIGURATION_DEBUG
    VerifyDecl(objc_method_decl);
#endif
    
    return objc_method_decl;
}

bool
ClangASTContext::GetHasExternalStorage (const CompilerType &type)
{
    if (IsClangType(type))
        return false;

    clang::QualType qual_type (GetCanonicalQualType(type));

    const clang::Type::TypeClass type_class = qual_type->getTypeClass();
    switch (type_class)
    {
        case clang::Type::Record:
        {
            clang::CXXRecordDecl *cxx_record_decl = qual_type->getAsCXXRecordDecl();
            if (cxx_record_decl)
                return cxx_record_decl->hasExternalLexicalStorage () || cxx_record_decl->hasExternalVisibleStorage();
        }
            break;

        case clang::Type::Enum:
        {
            clang::EnumDecl *enum_decl = llvm::cast<clang::EnumType>(qual_type)->getDecl();
            if (enum_decl)
                return enum_decl->hasExternalLexicalStorage () || enum_decl->hasExternalVisibleStorage();
        }
            break;

        case clang::Type::ObjCObject:
        case clang::Type::ObjCInterface:
        {
            const clang::ObjCObjectType *objc_class_type = llvm::dyn_cast<clang::ObjCObjectType>(qual_type.getTypePtr());
            assert (objc_class_type);
            if (objc_class_type)
            {
                clang::ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterface();

                if (class_interface_decl)
                    return class_interface_decl->hasExternalLexicalStorage () || class_interface_decl->hasExternalVisibleStorage ();
            }
        }
            break;

        case clang::Type::Typedef:
            return GetHasExternalStorage (CompilerType(type.GetTypeSystem(), llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType().getAsOpaquePtr()));

        case clang::Type::Auto:
            return GetHasExternalStorage (CompilerType(type.GetTypeSystem(), llvm::cast<clang::AutoType>(qual_type)->getDeducedType().getAsOpaquePtr()));
            
        case clang::Type::Elaborated:
            return GetHasExternalStorage (CompilerType(type.GetTypeSystem(), llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType().getAsOpaquePtr()));

        case clang::Type::Paren:
            return GetHasExternalStorage (CompilerType(type.GetTypeSystem(), llvm::cast<clang::ParenType>(qual_type)->desugar().getAsOpaquePtr()));

        default:
            break;
    }
    return false;
}


bool
ClangASTContext::SetHasExternalStorage (lldb::opaque_compiler_type_t type, bool has_extern)
{
    if (!type)
        return false;
    
    clang::QualType qual_type (GetCanonicalQualType(type));
    
    const clang::Type::TypeClass type_class = qual_type->getTypeClass();
    switch (type_class)
    {
        case clang::Type::Record:
        {
            clang::CXXRecordDecl *cxx_record_decl = qual_type->getAsCXXRecordDecl();
            if (cxx_record_decl)
            {
                cxx_record_decl->setHasExternalLexicalStorage (has_extern);
                cxx_record_decl->setHasExternalVisibleStorage (has_extern);
                return true;
            }
        }
            break;
            
        case clang::Type::Enum:
        {
            clang::EnumDecl *enum_decl = llvm::cast<clang::EnumType>(qual_type)->getDecl();
            if (enum_decl)
            {
                enum_decl->setHasExternalLexicalStorage (has_extern);
                enum_decl->setHasExternalVisibleStorage (has_extern);
                return true;
            }
        }
            break;
            
        case clang::Type::ObjCObject:
        case clang::Type::ObjCInterface:
        {
            const clang::ObjCObjectType *objc_class_type = llvm::dyn_cast<clang::ObjCObjectType>(qual_type.getTypePtr());
            assert (objc_class_type);
            if (objc_class_type)
            {
                clang::ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterface();
                
                if (class_interface_decl)
                {
                    class_interface_decl->setHasExternalLexicalStorage (has_extern);
                    class_interface_decl->setHasExternalVisibleStorage (has_extern);
                    return true;
                }
            }
        }
            break;
            
        case clang::Type::Typedef:
            return SetHasExternalStorage(llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType().getAsOpaquePtr(), has_extern);

        case clang::Type::Auto:
            return SetHasExternalStorage (llvm::cast<clang::AutoType>(qual_type)->getDeducedType().getAsOpaquePtr(), has_extern);
            
        case clang::Type::Elaborated:
            return SetHasExternalStorage (llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType().getAsOpaquePtr(), has_extern);
            
        case clang::Type::Paren:
            return SetHasExternalStorage (llvm::cast<clang::ParenType>(qual_type)->desugar().getAsOpaquePtr(), has_extern);
            
        default:
            break;
    }
    return false;
}


bool
ClangASTContext::CanImport (const CompilerType &type, lldb_private::ClangASTImporter &importer)
{
    if (IsClangType(type))
    {
        // TODO: remove external completion BOOL
        // CompleteAndFetchChildren should get the Decl out and check for the

        clang::QualType qual_type(GetCanonicalQualType(RemoveFastQualifiers(type)));

        const clang::Type::TypeClass type_class = qual_type->getTypeClass();
        switch (type_class)
        {
            case clang::Type::Record:
            {
                const clang::CXXRecordDecl *cxx_record_decl = qual_type->getAsCXXRecordDecl();
                if (cxx_record_decl)
                {
                    if (importer.ResolveDeclOrigin (cxx_record_decl, NULL, NULL))
                        return true;
                }
            }
                break;

            case clang::Type::Enum:
            {
                clang::EnumDecl *enum_decl = llvm::cast<clang::EnumType>(qual_type)->getDecl();
                if (enum_decl)
                {
                    if (importer.ResolveDeclOrigin (enum_decl, NULL, NULL))
                        return true;
                }
            }
                break;

            case clang::Type::ObjCObject:
            case clang::Type::ObjCInterface:
            {
                const clang::ObjCObjectType *objc_class_type = llvm::dyn_cast<clang::ObjCObjectType>(qual_type);
                if (objc_class_type)
                {
                    clang::ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterface();
                    // We currently can't complete objective C types through the newly added ASTContext
                    // because it only supports TagDecl objects right now...
                    if (class_interface_decl)
                    {
                        if (importer.ResolveDeclOrigin (class_interface_decl, NULL, NULL))
                            return true;
                    }
                }
            }
                break;


            case clang::Type::Typedef:
                return CanImport(CompilerType (type.GetTypeSystem(), llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType().getAsOpaquePtr()), importer);

            case clang::Type::Auto:
                return CanImport(CompilerType (type.GetTypeSystem(), llvm::cast<clang::AutoType>(qual_type)->getDeducedType().getAsOpaquePtr()), importer);
                
            case clang::Type::Elaborated:
                return CanImport(CompilerType (type.GetTypeSystem(), llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType().getAsOpaquePtr()), importer);

            case clang::Type::Paren:
                return CanImport(CompilerType (type.GetTypeSystem(), llvm::cast<clang::ParenType>(qual_type)->desugar().getAsOpaquePtr()), importer);

            default:
                break;
        }
    }
    return false;
}
bool
ClangASTContext::Import (const CompilerType &type, lldb_private::ClangASTImporter &importer)
{
    if (IsClangType(type))
    {
        // TODO: remove external completion BOOL
        // CompleteAndFetchChildren should get the Decl out and check for the

        clang::QualType qual_type(GetCanonicalQualType(RemoveFastQualifiers(type)));

        const clang::Type::TypeClass type_class = qual_type->getTypeClass();
        switch (type_class)
        {
            case clang::Type::Record:
            {
                const clang::CXXRecordDecl *cxx_record_decl = qual_type->getAsCXXRecordDecl();
                if (cxx_record_decl)
                {
                    if (importer.ResolveDeclOrigin (cxx_record_decl, NULL, NULL))
                        return importer.CompleteAndFetchChildren(qual_type);
                }
            }
                break;

            case clang::Type::Enum:
            {
                clang::EnumDecl *enum_decl = llvm::cast<clang::EnumType>(qual_type)->getDecl();
                if (enum_decl)
                {
                    if (importer.ResolveDeclOrigin (enum_decl, NULL, NULL))
                        return importer.CompleteAndFetchChildren(qual_type);
                }
            }
                break;

            case clang::Type::ObjCObject:
            case clang::Type::ObjCInterface:
            {
                const clang::ObjCObjectType *objc_class_type = llvm::dyn_cast<clang::ObjCObjectType>(qual_type);
                if (objc_class_type)
                {
                    clang::ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterface();
                    // We currently can't complete objective C types through the newly added ASTContext
                    // because it only supports TagDecl objects right now...
                    if (class_interface_decl)
                    {
                        if (importer.ResolveDeclOrigin (class_interface_decl, NULL, NULL))
                            return importer.CompleteAndFetchChildren(qual_type);
                    }
                }
            }
                break;


            case clang::Type::Typedef:
                return Import (CompilerType(type.GetTypeSystem(), llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType().getAsOpaquePtr()), importer);

            case clang::Type::Auto:
                return Import (CompilerType(type.GetTypeSystem(),llvm::cast<clang::AutoType>(qual_type)->getDeducedType().getAsOpaquePtr()), importer);
                
            case clang::Type::Elaborated:
                return Import (CompilerType(type.GetTypeSystem(),llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType().getAsOpaquePtr()), importer);

            case clang::Type::Paren:
                return Import (CompilerType(type.GetTypeSystem(),llvm::cast<clang::ParenType>(qual_type)->desugar().getAsOpaquePtr()), importer);
                
            default:
                break;
        }
    }
    return false;
}


#pragma mark TagDecl

bool
ClangASTContext::StartTagDeclarationDefinition (const CompilerType &type)
{
    clang::QualType qual_type (ClangASTContext::GetQualType(type));
    if (!qual_type.isNull())
    {
        const clang::TagType *tag_type = qual_type->getAs<clang::TagType>();
        if (tag_type)
        {
            clang::TagDecl *tag_decl = tag_type->getDecl();
            if (tag_decl)
            {
                tag_decl->startDefinition();
                return true;
            }
        }
        
        const clang::ObjCObjectType *object_type = qual_type->getAs<clang::ObjCObjectType>();
        if (object_type)
        {
            clang::ObjCInterfaceDecl *interface_decl = object_type->getInterface();
            if (interface_decl)
            {
                interface_decl->startDefinition();
                return true;
            }
        }
    }
    return false;
}

bool
ClangASTContext::CompleteTagDeclarationDefinition (const CompilerType& type)
{
    clang::QualType qual_type (ClangASTContext::GetQualType(type));
    if (!qual_type.isNull())
    {
        clang::CXXRecordDecl *cxx_record_decl = qual_type->getAsCXXRecordDecl();
        
        if (cxx_record_decl)
        {
            if (!cxx_record_decl->isCompleteDefinition())
                cxx_record_decl->completeDefinition();
            cxx_record_decl->setHasLoadedFieldsFromExternalStorage(true);
            cxx_record_decl->setHasExternalLexicalStorage (false);
            cxx_record_decl->setHasExternalVisibleStorage (false);
            return true;
        }
        
        const clang::EnumType *enutype = qual_type->getAs<clang::EnumType>();
        
        if (enutype)
        {
            clang::EnumDecl *enum_decl = enutype->getDecl();
            
            if (enum_decl)
            {
                if (!enum_decl->isCompleteDefinition())
                {
                    ClangASTContext *lldb_ast = llvm::dyn_cast<ClangASTContext>(type.GetTypeSystem());
                    if (lldb_ast == nullptr)
                        return false;
                    clang::ASTContext *ast = lldb_ast->getASTContext();

                    /// TODO This really needs to be fixed.
                    
                    unsigned NumPositiveBits = 1;
                    unsigned NumNegativeBits = 0;
                    
                    clang::QualType promotion_qual_type;
                    // If the enum integer type is less than an integer in bit width,
                    // then we must promote it to an integer size.
                    if (ast->getTypeSize(enum_decl->getIntegerType()) < ast->getTypeSize(ast->IntTy))
                    {
                        if (enum_decl->getIntegerType()->isSignedIntegerType())
                            promotion_qual_type = ast->IntTy;
                        else
                            promotion_qual_type = ast->UnsignedIntTy;
                    }
                    else
                        promotion_qual_type = enum_decl->getIntegerType();
                    
                    enum_decl->completeDefinition(enum_decl->getIntegerType(), promotion_qual_type, NumPositiveBits, NumNegativeBits);
                }
                return true;
            }
        }
    }
    return false;
}

bool
ClangASTContext::AddEnumerationValueToEnumerationType (lldb::opaque_compiler_type_t type,
                                                       const CompilerType &enumerator_clang_type,
                                                       const Declaration &decl,
                                                       const char *name,
                                                       int64_t enum_value,
                                                       uint32_t enum_value_bit_size)
{
    if (type && enumerator_clang_type.IsValid() && name && name[0])
    {
        clang::QualType enum_qual_type (GetCanonicalQualType(type));
        
        bool is_signed = false;
        enumerator_clang_type.IsIntegerType (is_signed);
        const clang::Type *clang_type = enum_qual_type.getTypePtr();
        if (clang_type)
        {
            const clang::EnumType *enutype = llvm::dyn_cast<clang::EnumType>(clang_type);
            
            if (enutype)
            {
                llvm::APSInt enum_llvm_apsint(enum_value_bit_size, is_signed);
                enum_llvm_apsint = enum_value;
                clang::EnumConstantDecl *enumerator_decl =
                clang::EnumConstantDecl::Create (*getASTContext(),
                                                 enutype->getDecl(),
                                                 clang::SourceLocation(),
                                                 name ? &getASTContext()->Idents.get(name) : nullptr,    // Identifier
                                                 GetQualType(enumerator_clang_type),
                                                 nullptr,
                                                 enum_llvm_apsint);
                
                if (enumerator_decl)
                {
                    enutype->getDecl()->addDecl(enumerator_decl);
                    
#ifdef LLDB_CONFIGURATION_DEBUG
                    VerifyDecl(enumerator_decl);
#endif
                    
                    return true;
                }
            }
        }
    }
    return false;
}

CompilerType
ClangASTContext::GetEnumerationIntegerType (lldb::opaque_compiler_type_t type)
{
    clang::QualType enum_qual_type (GetCanonicalQualType(type));
    const clang::Type *clang_type = enum_qual_type.getTypePtr();
    if (clang_type)
    {
        const clang::EnumType *enutype = llvm::dyn_cast<clang::EnumType>(clang_type);
        if (enutype)
        {
            clang::EnumDecl *enum_decl = enutype->getDecl();
            if (enum_decl)
                return CompilerType (getASTContext(), enum_decl->getIntegerType());
        }
    }
    return CompilerType();
}

CompilerType
ClangASTContext::CreateMemberPointerType (const CompilerType& type, const CompilerType &pointee_type)
{
    if (type && pointee_type.IsValid() && type.GetTypeSystem() == pointee_type.GetTypeSystem())
    {
        ClangASTContext *ast = llvm::dyn_cast<ClangASTContext>(type.GetTypeSystem());
        if (!ast)
            return CompilerType();
        return CompilerType (ast->getASTContext(),
                             ast->getASTContext()->getMemberPointerType (GetQualType(pointee_type),
                                                                         GetQualType(type).getTypePtr()));
    }
    return CompilerType();
}


size_t
ClangASTContext::ConvertStringToFloatValue (lldb::opaque_compiler_type_t type, const char *s, uint8_t *dst, size_t dst_size)
{
    if (type)
    {
        clang::QualType qual_type (GetCanonicalQualType(type));
        uint32_t count = 0;
        bool is_complex = false;
        if (IsFloatingPointType (type, count, is_complex))
        {
            // TODO: handle complex and vector types
            if (count != 1)
                return false;
            
            llvm::StringRef s_sref(s);
            llvm::APFloat ap_float(getASTContext()->getFloatTypeSemantics(qual_type), s_sref);
            
            const uint64_t bit_size = getASTContext()->getTypeSize (qual_type);
            const uint64_t byte_size = bit_size / 8;
            if (dst_size >= byte_size)
            {
                if (bit_size == sizeof(float)*8)
                {
                    float float32 = ap_float.convertToFloat();
                    ::memcpy (dst, &float32, byte_size);
                    return byte_size;
                }
                else if (bit_size >= 64)
                {
                    llvm::APInt ap_int(ap_float.bitcastToAPInt());
                    ::memcpy (dst, ap_int.getRawData(), byte_size);
                    return byte_size;
                }
            }
        }
    }
    return 0;
}



//----------------------------------------------------------------------
// Dumping types
//----------------------------------------------------------------------
#define DEPTH_INCREMENT 2

void
ClangASTContext::DumpValue (lldb::opaque_compiler_type_t type, ExecutionContext *exe_ctx,
                               Stream *s,
                               lldb::Format format,
                               const lldb_private::DataExtractor &data,
                               lldb::offset_t data_byte_offset,
                               size_t data_byte_size,
                               uint32_t bitfield_bit_size,
                               uint32_t bitfield_bit_offset,
                               bool show_types,
                               bool show_summary,
                               bool verbose,
                               uint32_t depth)
{
    if (!type)
        return;
    
    clang::QualType qual_type(GetQualType(type));
    switch (qual_type->getTypeClass())
    {
        case clang::Type::Record:
            if (GetCompleteType(type))
            {
                const clang::RecordType *record_type = llvm::cast<clang::RecordType>(qual_type.getTypePtr());
                const clang::RecordDecl *record_decl = record_type->getDecl();
                assert(record_decl);
                uint32_t field_bit_offset = 0;
                uint32_t field_byte_offset = 0;
                const clang::ASTRecordLayout &record_layout = getASTContext()->getASTRecordLayout(record_decl);
                uint32_t child_idx = 0;
                
                const clang::CXXRecordDecl *cxx_record_decl = llvm::dyn_cast<clang::CXXRecordDecl>(record_decl);
                if (cxx_record_decl)
                {
                    // We might have base classes to print out first
                    clang::CXXRecordDecl::base_class_const_iterator base_class, base_class_end;
                    for (base_class = cxx_record_decl->bases_begin(), base_class_end = cxx_record_decl->bases_end();
                         base_class != base_class_end;
                         ++base_class)
                    {
                        const clang::CXXRecordDecl *base_class_decl = llvm::cast<clang::CXXRecordDecl>(base_class->getType()->getAs<clang::RecordType>()->getDecl());
                        
                        // Skip empty base classes
                        if (verbose == false && ClangASTContext::RecordHasFields(base_class_decl) == false)
                            continue;
                        
                        if (base_class->isVirtual())
                            field_bit_offset = record_layout.getVBaseClassOffset(base_class_decl).getQuantity() * 8;
                        else
                            field_bit_offset = record_layout.getBaseClassOffset(base_class_decl).getQuantity() * 8;
                        field_byte_offset = field_bit_offset / 8;
                        assert (field_bit_offset % 8 == 0);
                        if (child_idx == 0)
                            s->PutChar('{');
                        else
                            s->PutChar(',');
                        
                        clang::QualType base_class_qual_type = base_class->getType();
                        std::string base_class_type_name(base_class_qual_type.getAsString());
                        
                        // Indent and print the base class type name
                        s->Printf("\n%*s%s ", depth + DEPTH_INCREMENT, "", base_class_type_name.c_str());
                        
                        clang::TypeInfo base_class_type_info = getASTContext()->getTypeInfo(base_class_qual_type);
                        
                        // Dump the value of the member
                        CompilerType base_clang_type(getASTContext(), base_class_qual_type);
                        base_clang_type.DumpValue (exe_ctx,
                                                   s,                                   // Stream to dump to
                                                   base_clang_type.GetFormat(),         // The format with which to display the member
                                                   data,                                // Data buffer containing all bytes for this type
                                                   data_byte_offset + field_byte_offset,// Offset into "data" where to grab value from
                                                   base_class_type_info.Width / 8,      // Size of this type in bytes
                                                   0,                                   // Bitfield bit size
                                                   0,                                   // Bitfield bit offset
                                                   show_types,                          // Boolean indicating if we should show the variable types
                                                   show_summary,                        // Boolean indicating if we should show a summary for the current type
                                                   verbose,                             // Verbose output?
                                                   depth + DEPTH_INCREMENT);            // Scope depth for any types that have children
                        
                        ++child_idx;
                    }
                }
                uint32_t field_idx = 0;
                clang::RecordDecl::field_iterator field, field_end;
                for (field = record_decl->field_begin(), field_end = record_decl->field_end(); field != field_end; ++field, ++field_idx, ++child_idx)
                {
                    // Print the starting squiggly bracket (if this is the
                    // first member) or comma (for member 2 and beyond) for
                    // the struct/union/class member.
                    if (child_idx == 0)
                        s->PutChar('{');
                    else
                        s->PutChar(',');
                    
                    // Indent
                    s->Printf("\n%*s", depth + DEPTH_INCREMENT, "");
                    
                    clang::QualType field_type = field->getType();
                    // Print the member type if requested
                    // Figure out the type byte size (field_type_info.first) and
                    // alignment (field_type_info.second) from the AST context.
                    clang::TypeInfo field_type_info = getASTContext()->getTypeInfo(field_type);
                    assert(field_idx < record_layout.getFieldCount());
                    // Figure out the field offset within the current struct/union/class type
                    field_bit_offset = record_layout.getFieldOffset (field_idx);
                    field_byte_offset = field_bit_offset / 8;
                    uint32_t field_bitfield_bit_size = 0;
                    uint32_t field_bitfield_bit_offset = 0;
                    if (ClangASTContext::FieldIsBitfield (getASTContext(), *field, field_bitfield_bit_size))
                        field_bitfield_bit_offset = field_bit_offset % 8;
                    
                    if (show_types)
                    {
                        std::string field_type_name(field_type.getAsString());
                        if (field_bitfield_bit_size > 0)
                            s->Printf("(%s:%u) ", field_type_name.c_str(), field_bitfield_bit_size);
                        else
                            s->Printf("(%s) ", field_type_name.c_str());
                    }
                    // Print the member name and equal sign
                    s->Printf("%s = ", field->getNameAsString().c_str());
                    
                    
                    // Dump the value of the member
                    CompilerType field_clang_type (getASTContext(), field_type);
                    field_clang_type.DumpValue (exe_ctx,
                                                s,                              // Stream to dump to
                                                field_clang_type.GetFormat(),   // The format with which to display the member
                                                data,                           // Data buffer containing all bytes for this type
                                                data_byte_offset + field_byte_offset,// Offset into "data" where to grab value from
                                                field_type_info.Width / 8,      // Size of this type in bytes
                                                field_bitfield_bit_size,        // Bitfield bit size
                                                field_bitfield_bit_offset,      // Bitfield bit offset
                                                show_types,                     // Boolean indicating if we should show the variable types
                                                show_summary,                   // Boolean indicating if we should show a summary for the current type
                                                verbose,                        // Verbose output?
                                                depth + DEPTH_INCREMENT);       // Scope depth for any types that have children
                }
                
                // Indent the trailing squiggly bracket
                if (child_idx > 0)
                    s->Printf("\n%*s}", depth, "");
            }
            return;
            
        case clang::Type::Enum:
            if (GetCompleteType(type))
            {
                const clang::EnumType *enutype = llvm::cast<clang::EnumType>(qual_type.getTypePtr());
                const clang::EnumDecl *enum_decl = enutype->getDecl();
                assert(enum_decl);
                clang::EnumDecl::enumerator_iterator enum_pos, enum_end_pos;
                lldb::offset_t offset = data_byte_offset;
                const int64_t enum_value = data.GetMaxU64Bitfield(&offset, data_byte_size, bitfield_bit_size, bitfield_bit_offset);
                for (enum_pos = enum_decl->enumerator_begin(), enum_end_pos = enum_decl->enumerator_end(); enum_pos != enum_end_pos; ++enum_pos)
                {
                    if (enum_pos->getInitVal() == enum_value)
                    {
                        s->Printf("%s", enum_pos->getNameAsString().c_str());
                        return;
                    }
                }
                // If we have gotten here we didn't get find the enumerator in the
                // enum decl, so just print the integer.
                s->Printf("%" PRIi64, enum_value);
            }
            return;
            
        case clang::Type::ConstantArray:
        {
            const clang::ConstantArrayType *array = llvm::cast<clang::ConstantArrayType>(qual_type.getTypePtr());
            bool is_array_of_characters = false;
            clang::QualType element_qual_type = array->getElementType();
            
            const clang::Type *canonical_type = element_qual_type->getCanonicalTypeInternal().getTypePtr();
            if (canonical_type)
                is_array_of_characters = canonical_type->isCharType();
            
            const uint64_t element_count = array->getSize().getLimitedValue();
            
            clang::TypeInfo field_type_info = getASTContext()->getTypeInfo(element_qual_type);
            
            uint32_t element_idx = 0;
            uint32_t element_offset = 0;
            uint64_t element_byte_size = field_type_info.Width / 8;
            uint32_t element_stride = element_byte_size;
            
            if (is_array_of_characters)
            {
                s->PutChar('"');
                data.Dump(s, data_byte_offset, lldb::eFormatChar, element_byte_size, element_count, UINT32_MAX, LLDB_INVALID_ADDRESS, 0, 0);
                s->PutChar('"');
                return;
            }
            else
            {
                CompilerType element_clang_type(getASTContext(), element_qual_type);
                lldb::Format element_format = element_clang_type.GetFormat();
                
                for (element_idx = 0; element_idx < element_count; ++element_idx)
                {
                    // Print the starting squiggly bracket (if this is the
                    // first member) or comman (for member 2 and beyong) for
                    // the struct/union/class member.
                    if (element_idx == 0)
                        s->PutChar('{');
                    else
                        s->PutChar(',');
                    
                    // Indent and print the index
                    s->Printf("\n%*s[%u] ", depth + DEPTH_INCREMENT, "", element_idx);
                    
                    // Figure out the field offset within the current struct/union/class type
                    element_offset = element_idx * element_stride;
                    
                    // Dump the value of the member
                    element_clang_type.DumpValue (exe_ctx,
                                                  s,                              // Stream to dump to
                                                  element_format,                 // The format with which to display the element
                                                  data,                           // Data buffer containing all bytes for this type
                                                  data_byte_offset + element_offset,// Offset into "data" where to grab value from
                                                  element_byte_size,              // Size of this type in bytes
                                                  0,                              // Bitfield bit size
                                                  0,                              // Bitfield bit offset
                                                  show_types,                     // Boolean indicating if we should show the variable types
                                                  show_summary,                   // Boolean indicating if we should show a summary for the current type
                                                  verbose,                        // Verbose output?
                                                  depth + DEPTH_INCREMENT);       // Scope depth for any types that have children
                }
                
                // Indent the trailing squiggly bracket
                if (element_idx > 0)
                    s->Printf("\n%*s}", depth, "");
            }
        }
            return;
            
        case clang::Type::Typedef:
        {
            clang::QualType typedef_qual_type = llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType();
            
            CompilerType typedef_clang_type (getASTContext(), typedef_qual_type);
            lldb::Format typedef_format = typedef_clang_type.GetFormat();
            clang::TypeInfo typedef_type_info = getASTContext()->getTypeInfo(typedef_qual_type);
            uint64_t typedef_byte_size = typedef_type_info.Width / 8;
            
            return typedef_clang_type.DumpValue (exe_ctx,
                                                 s,                  // Stream to dump to
                                                 typedef_format,     // The format with which to display the element
                                                 data,               // Data buffer containing all bytes for this type
                                                 data_byte_offset,   // Offset into "data" where to grab value from
                                                 typedef_byte_size,  // Size of this type in bytes
                                                 bitfield_bit_size,  // Bitfield bit size
                                                 bitfield_bit_offset,// Bitfield bit offset
                                                 show_types,         // Boolean indicating if we should show the variable types
                                                 show_summary,       // Boolean indicating if we should show a summary for the current type
                                                 verbose,            // Verbose output?
                                                 depth);             // Scope depth for any types that have children
        }
            break;
            
        case clang::Type::Auto:
        {
            clang::QualType elaborated_qual_type = llvm::cast<clang::AutoType>(qual_type)->getDeducedType();
            CompilerType elaborated_clang_type (getASTContext(), elaborated_qual_type);
            lldb::Format elaborated_format = elaborated_clang_type.GetFormat();
            clang::TypeInfo elaborated_type_info = getASTContext()->getTypeInfo(elaborated_qual_type);
            uint64_t elaborated_byte_size = elaborated_type_info.Width / 8;
            
            return elaborated_clang_type.DumpValue (exe_ctx,
                                                    s,                  // Stream to dump to
                                                    elaborated_format,  // The format with which to display the element
                                                    data,               // Data buffer containing all bytes for this type
                                                    data_byte_offset,   // Offset into "data" where to grab value from
                                                    elaborated_byte_size,  // Size of this type in bytes
                                                    bitfield_bit_size,  // Bitfield bit size
                                                    bitfield_bit_offset,// Bitfield bit offset
                                                    show_types,         // Boolean indicating if we should show the variable types
                                                    show_summary,       // Boolean indicating if we should show a summary for the current type
                                                    verbose,            // Verbose output?
                                                    depth);             // Scope depth for any types that have children
        }
            break;
            
        case clang::Type::Elaborated:
        {
            clang::QualType elaborated_qual_type = llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType();
            CompilerType elaborated_clang_type (getASTContext(), elaborated_qual_type);
            lldb::Format elaborated_format = elaborated_clang_type.GetFormat();
            clang::TypeInfo elaborated_type_info = getASTContext()->getTypeInfo(elaborated_qual_type);
            uint64_t elaborated_byte_size = elaborated_type_info.Width / 8;
            
            return elaborated_clang_type.DumpValue (exe_ctx,
                                                    s,                  // Stream to dump to
                                                    elaborated_format,  // The format with which to display the element
                                                    data,               // Data buffer containing all bytes for this type
                                                    data_byte_offset,   // Offset into "data" where to grab value from
                                                    elaborated_byte_size,  // Size of this type in bytes
                                                    bitfield_bit_size,  // Bitfield bit size
                                                    bitfield_bit_offset,// Bitfield bit offset
                                                    show_types,         // Boolean indicating if we should show the variable types
                                                    show_summary,       // Boolean indicating if we should show a summary for the current type
                                                    verbose,            // Verbose output?
                                                    depth);             // Scope depth for any types that have children
        }
            break;
            
        case clang::Type::Paren:
        {
            clang::QualType desugar_qual_type = llvm::cast<clang::ParenType>(qual_type)->desugar();
            CompilerType desugar_clang_type (getASTContext(), desugar_qual_type);
            
            lldb::Format desugar_format = desugar_clang_type.GetFormat();
            clang::TypeInfo desugar_type_info = getASTContext()->getTypeInfo(desugar_qual_type);
            uint64_t desugar_byte_size = desugar_type_info.Width / 8;
            
            return desugar_clang_type.DumpValue (exe_ctx,
                                                 s,                  // Stream to dump to
                                                 desugar_format,  // The format with which to display the element
                                                 data,               // Data buffer containing all bytes for this type
                                                 data_byte_offset,   // Offset into "data" where to grab value from
                                                 desugar_byte_size,  // Size of this type in bytes
                                                 bitfield_bit_size,  // Bitfield bit size
                                                 bitfield_bit_offset,// Bitfield bit offset
                                                 show_types,         // Boolean indicating if we should show the variable types
                                                 show_summary,       // Boolean indicating if we should show a summary for the current type
                                                 verbose,            // Verbose output?
                                                 depth);             // Scope depth for any types that have children
        }
            break;
            
        default:
            // We are down to a scalar type that we just need to display.
            data.Dump(s,
                      data_byte_offset,
                      format,
                      data_byte_size,
                      1,
                      UINT32_MAX,
                      LLDB_INVALID_ADDRESS,
                      bitfield_bit_size,
                      bitfield_bit_offset);
            
            if (show_summary)
                DumpSummary (type, exe_ctx, s, data, data_byte_offset, data_byte_size);
            break;
    }
}




bool
ClangASTContext::DumpTypeValue (lldb::opaque_compiler_type_t type, Stream *s,
                                   lldb::Format format,
                                   const lldb_private::DataExtractor &data,
                                   lldb::offset_t byte_offset,
                                   size_t byte_size,
                                   uint32_t bitfield_bit_size,
                                   uint32_t bitfield_bit_offset,
                                   ExecutionContextScope *exe_scope)
{
    if (!type)
        return false;
    if (IsAggregateType(type))
    {
        return false;
    }
    else
    {
        clang::QualType qual_type(GetQualType(type));
        
        const clang::Type::TypeClass type_class = qual_type->getTypeClass();
        switch (type_class)
        {
            case clang::Type::Typedef:
            {
                clang::QualType typedef_qual_type = llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getUnderlyingType();
                CompilerType typedef_clang_type (getASTContext(), typedef_qual_type);
                if (format == eFormatDefault)
                    format = typedef_clang_type.GetFormat();
                clang::TypeInfo typedef_type_info = getASTContext()->getTypeInfo(typedef_qual_type);
                uint64_t typedef_byte_size = typedef_type_info.Width / 8;
                
                return typedef_clang_type.DumpTypeValue (s,
                                                         format,                 // The format with which to display the element
                                                         data,                   // Data buffer containing all bytes for this type
                                                         byte_offset,            // Offset into "data" where to grab value from
                                                         typedef_byte_size,      // Size of this type in bytes
                                                         bitfield_bit_size,      // Size in bits of a bitfield value, if zero don't treat as a bitfield
                                                         bitfield_bit_offset,    // Offset in bits of a bitfield value if bitfield_bit_size != 0
                                                         exe_scope);
            }
                break;
                
            case clang::Type::Enum:
                // If our format is enum or default, show the enumeration value as
                // its enumeration string value, else just display it as requested.
                if ((format == eFormatEnum || format == eFormatDefault) && GetCompleteType(type))
                {
                    const clang::EnumType *enutype = llvm::cast<clang::EnumType>(qual_type.getTypePtr());
                    const clang::EnumDecl *enum_decl = enutype->getDecl();
                    assert(enum_decl);
                    clang::EnumDecl::enumerator_iterator enum_pos, enum_end_pos;
                    const bool is_signed = qual_type->isSignedIntegerOrEnumerationType();
                    lldb::offset_t offset = byte_offset;
                    if (is_signed)
                    {
                        const int64_t enum_svalue = data.GetMaxS64Bitfield (&offset, byte_size, bitfield_bit_size, bitfield_bit_offset);
                        for (enum_pos = enum_decl->enumerator_begin(), enum_end_pos = enum_decl->enumerator_end(); enum_pos != enum_end_pos; ++enum_pos)
                        {
                            if (enum_pos->getInitVal().getSExtValue() == enum_svalue)
                            {
                                s->PutCString (enum_pos->getNameAsString().c_str());
                                return true;
                            }
                        }
                        // If we have gotten here we didn't get find the enumerator in the
                        // enum decl, so just print the integer.
                        s->Printf("%" PRIi64, enum_svalue);
                    }
                    else
                    {
                        const uint64_t enum_uvalue = data.GetMaxU64Bitfield (&offset, byte_size, bitfield_bit_size, bitfield_bit_offset);
                        for (enum_pos = enum_decl->enumerator_begin(), enum_end_pos = enum_decl->enumerator_end(); enum_pos != enum_end_pos; ++enum_pos)
                        {
                            if (enum_pos->getInitVal().getZExtValue() == enum_uvalue)
                            {
                                s->PutCString (enum_pos->getNameAsString().c_str());
                                return true;
                            }
                        }
                        // If we have gotten here we didn't get find the enumerator in the
                        // enum decl, so just print the integer.
                        s->Printf("%" PRIu64, enum_uvalue);
                    }
                    return true;
                }
                // format was not enum, just fall through and dump the value as requested....
                
            default:
                // We are down to a scalar type that we just need to display.
            {
                uint32_t item_count = 1;
                // A few formats, we might need to modify our size and count for depending
                // on how we are trying to display the value...
                switch (format)
                {
                    default:
                    case eFormatBoolean:
                    case eFormatBinary:
                    case eFormatComplex:
                    case eFormatCString:         // NULL terminated C strings
                    case eFormatDecimal:
                    case eFormatEnum:
                    case eFormatHex:
                    case eFormatHexUppercase:
                    case eFormatFloat:
                    case eFormatOctal:
                    case eFormatOSType:
                    case eFormatUnsigned:
                    case eFormatPointer:
                    case eFormatVectorOfChar:
                    case eFormatVectorOfSInt8:
                    case eFormatVectorOfUInt8:
                    case eFormatVectorOfSInt16:
                    case eFormatVectorOfUInt16:
                    case eFormatVectorOfSInt32:
                    case eFormatVectorOfUInt32:
                    case eFormatVectorOfSInt64:
                    case eFormatVectorOfUInt64:
                    case eFormatVectorOfFloat32:
                    case eFormatVectorOfFloat64:
                    case eFormatVectorOfUInt128:
                        break;
                        
                    case eFormatChar:
                    case eFormatCharPrintable:
                    case eFormatCharArray:
                    case eFormatBytes:
                    case eFormatBytesWithASCII:
                        item_count = byte_size;
                        byte_size = 1;
                        break;
                        
                    case eFormatUnicode16:
                        item_count = byte_size / 2;
                        byte_size = 2;
                        break;
                        
                    case eFormatUnicode32:
                        item_count = byte_size / 4;
                        byte_size = 4;
                        break;
                }
                return data.Dump (s,
                                  byte_offset,
                                  format,
                                  byte_size,
                                  item_count,
                                  UINT32_MAX,
                                  LLDB_INVALID_ADDRESS,
                                  bitfield_bit_size,
                                  bitfield_bit_offset,
                                  exe_scope);
            }
                break;
        }
    }
    return 0;
}



void
ClangASTContext::DumpSummary (lldb::opaque_compiler_type_t type, ExecutionContext *exe_ctx,
                                 Stream *s,
                                 const lldb_private::DataExtractor &data,
                                 lldb::offset_t data_byte_offset,
                                 size_t data_byte_size)
{
    uint32_t length = 0;
    if (IsCStringType (type, length))
    {
        if (exe_ctx)
        {
            Process *process = exe_ctx->GetProcessPtr();
            if (process)
            {
                lldb::offset_t offset = data_byte_offset;
                lldb::addr_t pointer_address = data.GetMaxU64(&offset, data_byte_size);
                std::vector<uint8_t> buf;
                if (length > 0)
                    buf.resize (length);
                else
                    buf.resize (256);
                
                lldb_private::DataExtractor cstr_data(&buf.front(), buf.size(), process->GetByteOrder(), 4);
                buf.back() = '\0';
                size_t bytes_read;
                size_t total_cstr_len = 0;
                Error error;
                while ((bytes_read = process->ReadMemory (pointer_address, &buf.front(), buf.size(), error)) > 0)
                {
                    const size_t len = strlen((const char *)&buf.front());
                    if (len == 0)
                        break;
                    if (total_cstr_len == 0)
                        s->PutCString (" \"");
                    cstr_data.Dump(s, 0, lldb::eFormatChar, 1, len, UINT32_MAX, LLDB_INVALID_ADDRESS, 0, 0);
                    total_cstr_len += len;
                    if (len < buf.size())
                        break;
                    pointer_address += total_cstr_len;
                }
                if (total_cstr_len > 0)
                    s->PutChar ('"');
            }
        }
    }
}

void
ClangASTContext::DumpTypeDescription (lldb::opaque_compiler_type_t type)
{
    StreamFile s (stdout, false);
    DumpTypeDescription (type, &s);
    ClangASTMetadata *metadata = ClangASTContext::GetMetadata (getASTContext(), type);
    if (metadata)
    {
        metadata->Dump (&s);
    }
}

void
ClangASTContext::DumpTypeDescription (lldb::opaque_compiler_type_t type, Stream *s)
{
    if (type)
    {
        clang::QualType qual_type(GetQualType(type));
        
        llvm::SmallVector<char, 1024> buf;
        llvm::raw_svector_ostream llvm_ostrm (buf);
        
        const clang::Type::TypeClass type_class = qual_type->getTypeClass();
        switch (type_class)
        {
            case clang::Type::ObjCObject:
            case clang::Type::ObjCInterface:
            {
                GetCompleteType(type);
                
                const clang::ObjCObjectType *objc_class_type = llvm::dyn_cast<clang::ObjCObjectType>(qual_type.getTypePtr());
                assert (objc_class_type);
                if (objc_class_type)
                {
                    clang::ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterface();
                    if (class_interface_decl)
                    {
                        clang::PrintingPolicy policy = getASTContext()->getPrintingPolicy();
                        class_interface_decl->print(llvm_ostrm, policy, s->GetIndentLevel());
                    }
                }
            }
                break;
                
            case clang::Type::Typedef:
            {
                const clang::TypedefType *typedef_type = qual_type->getAs<clang::TypedefType>();
                if (typedef_type)
                {
                    const clang::TypedefNameDecl *typedef_decl = typedef_type->getDecl();
                    std::string clang_typedef_name (typedef_decl->getQualifiedNameAsString());
                    if (!clang_typedef_name.empty())
                    {
                        s->PutCString ("typedef ");
                        s->PutCString (clang_typedef_name.c_str());
                    }
                }
            }
                break;

            case clang::Type::Auto:
                CompilerType (getASTContext(), llvm::cast<clang::AutoType>(qual_type)->getDeducedType()).DumpTypeDescription(s);
                return;
                
            case clang::Type::Elaborated:
                CompilerType (getASTContext(), llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType()).DumpTypeDescription(s);
                return;
                
            case clang::Type::Paren:
                CompilerType (getASTContext(), llvm::cast<clang::ParenType>(qual_type)->desugar()).DumpTypeDescription(s);
                return;
                
            case clang::Type::Record:
            {
                GetCompleteType(type);
                
                const clang::RecordType *record_type = llvm::cast<clang::RecordType>(qual_type.getTypePtr());
                const clang::RecordDecl *record_decl = record_type->getDecl();
                const clang::CXXRecordDecl *cxx_record_decl = llvm::dyn_cast<clang::CXXRecordDecl>(record_decl);
                
                if (cxx_record_decl)
                    cxx_record_decl->print(llvm_ostrm, getASTContext()->getPrintingPolicy(), s->GetIndentLevel());
                else
                    record_decl->print(llvm_ostrm, getASTContext()->getPrintingPolicy(), s->GetIndentLevel());
            }
                break;
                
            default:
            {
                const clang::TagType *tag_type = llvm::dyn_cast<clang::TagType>(qual_type.getTypePtr());
                if (tag_type)
                {
                    clang::TagDecl *tag_decl = tag_type->getDecl();
                    if (tag_decl)
                        tag_decl->print(llvm_ostrm, 0);
                }
                else
                {
                    std::string clang_type_name(qual_type.getAsString());
                    if (!clang_type_name.empty())
                        s->PutCString (clang_type_name.c_str());
                }
            }
        }
        
        if (buf.size() > 0)
        {
            s->Write (buf.data(), buf.size());
        }
    }
}

void
ClangASTContext::DumpTypeName (const CompilerType &type)
{
    if (IsClangType(type))
    {
        clang::QualType qual_type(GetCanonicalQualType(RemoveFastQualifiers(type)));

        const clang::Type::TypeClass type_class = qual_type->getTypeClass();
        switch (type_class)
        {
            case clang::Type::Record:
            {
                const clang::CXXRecordDecl *cxx_record_decl = qual_type->getAsCXXRecordDecl();
                if (cxx_record_decl)
                    printf("class %s", cxx_record_decl->getName().str().c_str());
            }
                break;

            case clang::Type::Enum:
            {
                clang::EnumDecl *enum_decl = llvm::cast<clang::EnumType>(qual_type)->getDecl();
                if (enum_decl)
                {
                    printf("enum %s", enum_decl->getName().str().c_str());
                }
            }
                break;

            case clang::Type::ObjCObject:
            case clang::Type::ObjCInterface:
            {
                const clang::ObjCObjectType *objc_class_type = llvm::dyn_cast<clang::ObjCObjectType>(qual_type);
                if (objc_class_type)
                {
                    clang::ObjCInterfaceDecl *class_interface_decl = objc_class_type->getInterface();
                    // We currently can't complete objective C types through the newly added ASTContext
                    // because it only supports TagDecl objects right now...
                    if (class_interface_decl)
                        printf("@class %s", class_interface_decl->getName().str().c_str());
                }
            }
                break;


            case clang::Type::Typedef:
                printf("typedef %s", llvm::cast<clang::TypedefType>(qual_type)->getDecl()->getName().str().c_str());
                break;

            case clang::Type::Auto:
                printf("auto ");
                return DumpTypeName (CompilerType (type.GetTypeSystem(), llvm::cast<clang::AutoType>(qual_type)->getDeducedType().getAsOpaquePtr()));
                
            case clang::Type::Elaborated:
                printf("elaborated ");
                return DumpTypeName (CompilerType (type.GetTypeSystem(), llvm::cast<clang::ElaboratedType>(qual_type)->getNamedType().getAsOpaquePtr()));

            case clang::Type::Paren:
                printf("paren ");
                return DumpTypeName (CompilerType (type.GetTypeSystem(), llvm::cast<clang::ParenType>(qual_type)->desugar().getAsOpaquePtr()));

            default:
                printf("ClangASTContext::DumpTypeName() type_class = %u", type_class);
                break;
        }
    }

}



clang::ClassTemplateDecl *
ClangASTContext::ParseClassTemplateDecl (clang::DeclContext *decl_ctx,
                                         lldb::AccessType access_type,
                                         const char *parent_name,
                                         int tag_decl_kind,
                                         const ClangASTContext::TemplateParameterInfos &template_param_infos)
{
    if (template_param_infos.IsValid())
    {
        std::string template_basename(parent_name);
        template_basename.erase (template_basename.find('<'));

        return CreateClassTemplateDecl (decl_ctx,
                                        access_type,
                                        template_basename.c_str(),
                                        tag_decl_kind,
                                        template_param_infos);
    }
    return NULL;
}

void
ClangASTContext::CompleteTagDecl (void *baton, clang::TagDecl *decl)
{
    ClangASTContext *ast = (ClangASTContext *)baton;
    SymbolFile *sym_file = ast->GetSymbolFile();
    if (sym_file)
    {
        CompilerType clang_type = GetTypeForDecl (decl);
        if (clang_type)
            sym_file->CompleteType (clang_type);
    }
}

void
ClangASTContext::CompleteObjCInterfaceDecl (void *baton, clang::ObjCInterfaceDecl *decl)
{
    ClangASTContext *ast = (ClangASTContext *)baton;
    SymbolFile *sym_file = ast->GetSymbolFile();
    if (sym_file)
    {
        CompilerType clang_type = GetTypeForDecl (decl);
        if (clang_type)
            sym_file->CompleteType (clang_type);
    }
}


DWARFASTParser *
ClangASTContext::GetDWARFParser ()
{
    if (!m_dwarf_ast_parser_ap)
        m_dwarf_ast_parser_ap.reset(new DWARFASTParserClang(*this));
    return m_dwarf_ast_parser_ap.get();
}


bool
ClangASTContext::LayoutRecordType(void *baton,
                                  const clang::RecordDecl *record_decl,
                                  uint64_t &bit_size,
                                  uint64_t &alignment,
                                  llvm::DenseMap<const clang::FieldDecl *, uint64_t> &field_offsets,
                                  llvm::DenseMap<const clang::CXXRecordDecl *, clang::CharUnits> &base_offsets,
                                  llvm::DenseMap<const clang::CXXRecordDecl *, clang::CharUnits> &vbase_offsets)
{
    ClangASTContext *ast = (ClangASTContext *)baton;
    DWARFASTParserClang *dwarf_ast_parser = (DWARFASTParserClang *)ast->GetDWARFParser();
    return dwarf_ast_parser->LayoutRecordType(record_decl, bit_size, alignment, field_offsets, base_offsets, vbase_offsets);
}

//----------------------------------------------------------------------
// CompilerDecl override functions
//----------------------------------------------------------------------
lldb::VariableSP
ClangASTContext::DeclGetVariable (void *opaque_decl)
{
    if (llvm::dyn_cast<clang::VarDecl>((clang::Decl *)opaque_decl))
    {
        auto decl_search_it = m_decl_objects.find(opaque_decl);
        if (decl_search_it != m_decl_objects.end())
            return std::static_pointer_cast<Variable>(decl_search_it->second);
    }
    return VariableSP();
}

void
ClangASTContext::DeclLinkToObject (void *opaque_decl, std::shared_ptr<void> object)
{
    if (m_decl_objects.find(opaque_decl) == m_decl_objects.end())
        m_decl_objects.insert(std::make_pair(opaque_decl, object));
}

ConstString
ClangASTContext::DeclGetName (void *opaque_decl)
{
    if (opaque_decl)
    {
        clang::NamedDecl *nd = llvm::dyn_cast<NamedDecl>((clang::Decl*)opaque_decl);
        if (nd != nullptr)
            return ConstString(nd->getDeclName().getAsString());
    }
    return ConstString();
}

ConstString
ClangASTContext::DeclGetMangledName (void *opaque_decl)
{
    if (opaque_decl)
    {
        clang::NamedDecl *nd = llvm::dyn_cast<clang::NamedDecl>((clang::Decl*)opaque_decl);
        if (nd != nullptr && !llvm::isa<clang::ObjCMethodDecl>(nd))
        {
            clang::MangleContext *mc = getMangleContext();
            if (mc && mc->shouldMangleCXXName(nd))
            {
                llvm::SmallVector<char, 1024> buf;
                llvm::raw_svector_ostream llvm_ostrm (buf);
                if (llvm::isa<clang::CXXConstructorDecl>(nd))
                {
                    mc->mangleCXXCtor(llvm::dyn_cast<clang::CXXConstructorDecl>(nd), Ctor_Complete, llvm_ostrm);
                }
                else if (llvm::isa<clang::CXXDestructorDecl>(nd))
                {
                    mc->mangleCXXDtor(llvm::dyn_cast<clang::CXXDestructorDecl>(nd), Dtor_Complete, llvm_ostrm);
                }
                else
                {
                    mc->mangleName(nd, llvm_ostrm);
                }
                if (buf.size() > 0)
                    return ConstString(buf.data(), buf.size());
            }
        }
    }
    return ConstString();
}

CompilerDeclContext
ClangASTContext::DeclGetDeclContext (void *opaque_decl)
{
    if (opaque_decl)
        return CompilerDeclContext(this, ((clang::Decl*)opaque_decl)->getDeclContext());
    else
        return CompilerDeclContext();
}

CompilerType
ClangASTContext::DeclGetFunctionReturnType(void *opaque_decl)
{
    if (clang::FunctionDecl *func_decl = llvm::dyn_cast<clang::FunctionDecl>((clang::Decl*)opaque_decl))
        return CompilerType(this, func_decl->getReturnType().getAsOpaquePtr());
    if (clang::ObjCMethodDecl *objc_method = llvm::dyn_cast<clang::ObjCMethodDecl>((clang::Decl*)opaque_decl))
        return CompilerType(this, objc_method->getReturnType().getAsOpaquePtr());
    else
        return CompilerType();
}

size_t
ClangASTContext::DeclGetFunctionNumArguments(void *opaque_decl)
{
    if (clang::FunctionDecl *func_decl = llvm::dyn_cast<clang::FunctionDecl>((clang::Decl*)opaque_decl))
        return func_decl->param_size();
    if (clang::ObjCMethodDecl *objc_method = llvm::dyn_cast<clang::ObjCMethodDecl>((clang::Decl*)opaque_decl))
        return  objc_method->param_size();
    else
        return 0;
}

CompilerType
ClangASTContext::DeclGetFunctionArgumentType (void *opaque_decl, size_t idx)
{
    if (clang::FunctionDecl *func_decl = llvm::dyn_cast<clang::FunctionDecl>((clang::Decl*)opaque_decl))
    {
        if (idx < func_decl->param_size())
        {
            ParmVarDecl *var_decl = func_decl->getParamDecl(idx);
            if (var_decl)
                return  CompilerType(this, var_decl->getOriginalType().getAsOpaquePtr());
        }
    }
    else if (clang::ObjCMethodDecl *objc_method = llvm::dyn_cast<clang::ObjCMethodDecl>((clang::Decl*)opaque_decl))
    {
        if (idx < objc_method->param_size())
            return CompilerType(this, objc_method->parameters()[idx]->getOriginalType().getAsOpaquePtr());
    }
    return CompilerType();
}

//----------------------------------------------------------------------
// CompilerDeclContext functions
//----------------------------------------------------------------------

std::vector<CompilerDecl>
ClangASTContext::DeclContextFindDeclByName(void *opaque_decl_ctx, ConstString name)
{
    std::vector<CompilerDecl> found_decls;
    if (opaque_decl_ctx)
    {
        DeclContext *root_decl_ctx = (DeclContext *)opaque_decl_ctx;
        std::set<DeclContext *> searched;
        std::multimap<DeclContext *, DeclContext *> search_queue;
        SymbolFile *symbol_file = GetSymbolFile();

        for (clang::DeclContext *decl_context = root_decl_ctx; decl_context != nullptr && found_decls.empty(); decl_context = decl_context->getParent())
        {
            search_queue.insert(std::make_pair(decl_context, decl_context));

            for (auto it = search_queue.find(decl_context); it != search_queue.end(); it++)
            {
                if (!searched.insert(it->second).second)
                    continue;
                symbol_file->ParseDeclsForContext(CompilerDeclContext(this, it->second));

                for (clang::Decl *child : it->second->decls())
                {
                    if (clang::UsingDirectiveDecl *ud = llvm::dyn_cast<clang::UsingDirectiveDecl>(child))
                    {
                        clang::DeclContext *from = ud->getCommonAncestor();
                        if (searched.find(ud->getNominatedNamespace()) == searched.end())
                            search_queue.insert(std::make_pair(from, ud->getNominatedNamespace()));
                    }
                    else if (clang::UsingDecl *ud = llvm::dyn_cast<clang::UsingDecl>(child))
                    {
                        for (clang::UsingShadowDecl *usd : ud->shadows())
                        {
                            clang::Decl *target = usd->getTargetDecl();
                            if (clang::NamedDecl *nd = llvm::dyn_cast<clang::NamedDecl>(target))
                            {
                                IdentifierInfo *ii = nd->getIdentifier();
                                if (ii != nullptr && ii->getName().equals(name.AsCString(nullptr)))
                                    found_decls.push_back(CompilerDecl(this, nd));
                            }
                        }
                    }
                    else if (clang::NamedDecl *nd = llvm::dyn_cast<clang::NamedDecl>(child))
                    {
                        IdentifierInfo *ii = nd->getIdentifier();
                        if (ii != nullptr && ii->getName().equals(name.AsCString(nullptr)))
                            found_decls.push_back(CompilerDecl(this, nd));
                    }
                }
            }
        }
    }
    return found_decls;
}

// Look for child_decl_ctx's lookup scope in frame_decl_ctx and its parents,
// and return the number of levels it took to find it, or LLDB_INVALID_DECL_LEVEL
// if not found.  If the decl was imported via a using declaration, its name and/or
// type, if set, will be used to check that the decl found in the scope is a match.
//
// The optional name is required by languages (like C++) to handle using declarations
// like:
//
//     void poo();
//     namespace ns {
//         void foo();
//         void goo();
//     }
//     void bar() {
//         using ns::foo;
//         // CountDeclLevels returns 0 for 'foo', 1 for 'poo', and
//         // LLDB_INVALID_DECL_LEVEL for 'goo'.
//     }
//
// The optional type is useful in the case that there's a specific overload
// that we're looking for that might otherwise be shadowed, like:
//
//     void foo(int);
//     namespace ns {
//         void foo();
//     }
//     void bar() {
//         using ns::foo;
//         // CountDeclLevels returns 0 for { 'foo', void() },
//         // 1 for { 'foo', void(int) }, and
//         // LLDB_INVALID_DECL_LEVEL for { 'foo', void(int, int) }.
//     }
//
// NOTE: Because file statics are at the TranslationUnit along with globals, a
// function at file scope will return the same level as a function at global scope.
// Ideally we'd like to treat the file scope as an additional scope just below the
// global scope.  More work needs to be done to recognise that, if the decl we're
// trying to look up is static, we should compare its source file with that of the
// current scope and return a lower number for it.
uint32_t
ClangASTContext::CountDeclLevels (clang::DeclContext *frame_decl_ctx,
                                  clang::DeclContext *child_decl_ctx,
                                  ConstString *child_name,
                                  CompilerType *child_type)
{
    if (frame_decl_ctx)
    {
        std::set<DeclContext *> searched;
        std::multimap<DeclContext *, DeclContext *> search_queue;
        SymbolFile *symbol_file = GetSymbolFile();

        // Get the lookup scope for the decl we're trying to find.
        clang::DeclContext *parent_decl_ctx = child_decl_ctx->getParent();

        // Look for it in our scope's decl context and its parents.
        uint32_t level = 0;
        for (clang::DeclContext *decl_ctx = frame_decl_ctx; decl_ctx != nullptr; decl_ctx = decl_ctx->getParent())
        {
            if (!decl_ctx->isLookupContext())
                continue;
            if (decl_ctx == parent_decl_ctx)
                // Found it!
                return level;
            search_queue.insert(std::make_pair(decl_ctx, decl_ctx));
            for (auto it = search_queue.find(decl_ctx); it != search_queue.end(); it++)
            {
                if (searched.find(it->second) != searched.end())
                    continue;
                searched.insert(it->second);
                symbol_file->ParseDeclsForContext(CompilerDeclContext(this, it->second));

                for (clang::Decl *child : it->second->decls())
                {
                    if (clang::UsingDirectiveDecl *ud = llvm::dyn_cast<clang::UsingDirectiveDecl>(child))
                    {
                        clang::DeclContext *ns = ud->getNominatedNamespace();
                        if (ns == parent_decl_ctx)
                            // Found it!
                            return level;
                        clang::DeclContext *from = ud->getCommonAncestor();
                        if (searched.find(ns) == searched.end())
                            search_queue.insert(std::make_pair(from, ns));
                    }
                    else if (child_name)
                    {
                        if (clang::UsingDecl *ud = llvm::dyn_cast<clang::UsingDecl>(child))
                        {
                            for (clang::UsingShadowDecl *usd : ud->shadows())
                            {
                                clang::Decl *target = usd->getTargetDecl();
                                clang::NamedDecl *nd = llvm::dyn_cast<clang::NamedDecl>(target);
                                if (!nd)
                                    continue;
                                // Check names.
                                IdentifierInfo *ii = nd->getIdentifier();
                                if (ii == nullptr || !ii->getName().equals(child_name->AsCString(nullptr)))
                                    continue;
                                // Check types, if one was provided.
                                if (child_type)
                                {
                                    CompilerType clang_type = ClangASTContext::GetTypeForDecl(nd);
                                    if (!AreTypesSame(clang_type, *child_type, /*ignore_qualifiers=*/true))
                                        continue;
                                }
                                // Found it!
                                return level;
                            }
                        }
                    }
                }
            }
            ++level;
        }
    }
    return LLDB_INVALID_DECL_LEVEL;
}

bool
ClangASTContext::DeclContextIsStructUnionOrClass (void *opaque_decl_ctx)
{
    if (opaque_decl_ctx)
        return ((clang::DeclContext *)opaque_decl_ctx)->isRecord();
    else
        return false;
}

ConstString
ClangASTContext::DeclContextGetName (void *opaque_decl_ctx)
{
    if (opaque_decl_ctx)
    {
        clang::NamedDecl *named_decl = llvm::dyn_cast<clang::NamedDecl>((clang::DeclContext *)opaque_decl_ctx);
        if (named_decl)
            return ConstString(named_decl->getName());
    }
    return ConstString();
}

bool
ClangASTContext::DeclContextIsClassMethod (void *opaque_decl_ctx,
                                           lldb::LanguageType *language_ptr,
                                           bool *is_instance_method_ptr,
                                           ConstString *language_object_name_ptr)
{
    if (opaque_decl_ctx)
    {
        clang::DeclContext *decl_ctx = (clang::DeclContext *)opaque_decl_ctx;
        if (ObjCMethodDecl *objc_method = llvm::dyn_cast<clang::ObjCMethodDecl>(decl_ctx))
        {
            if (is_instance_method_ptr)
                *is_instance_method_ptr = objc_method->isInstanceMethod();
            if (language_ptr)
                *language_ptr = eLanguageTypeObjC;
            if (language_object_name_ptr)
                language_object_name_ptr->SetCString("self");
            return true;
        }
        else if (CXXMethodDecl *cxx_method = llvm::dyn_cast<clang::CXXMethodDecl>(decl_ctx))
        {
            if (is_instance_method_ptr)
                *is_instance_method_ptr = cxx_method->isInstance();
            if (language_ptr)
                *language_ptr = eLanguageTypeC_plus_plus;
            if (language_object_name_ptr)
                language_object_name_ptr->SetCString("this");
            return true;
        }
        else if (clang::FunctionDecl *function_decl = llvm::dyn_cast<clang::FunctionDecl>(decl_ctx))
        {
            ClangASTMetadata *metadata = GetMetadata (&decl_ctx->getParentASTContext(), function_decl);
            if (metadata && metadata->HasObjectPtr())
            {
                if (is_instance_method_ptr)
                    *is_instance_method_ptr = true;
                if (language_ptr)
                    *language_ptr = eLanguageTypeObjC;
                if (language_object_name_ptr)
                    language_object_name_ptr->SetCString (metadata->GetObjectPtrName());
                return true;
            }
        }
    }
    return false;
}

clang::DeclContext *
ClangASTContext::DeclContextGetAsDeclContext (const CompilerDeclContext &dc)
{
    if (dc.IsClang())
        return (clang::DeclContext *)dc.GetOpaqueDeclContext();
    return nullptr;
}


ObjCMethodDecl *
ClangASTContext::DeclContextGetAsObjCMethodDecl (const CompilerDeclContext &dc)
{
    if (dc.IsClang())
        return llvm::dyn_cast<clang::ObjCMethodDecl>((clang::DeclContext *)dc.GetOpaqueDeclContext());
    return nullptr;
}

CXXMethodDecl *
ClangASTContext::DeclContextGetAsCXXMethodDecl (const CompilerDeclContext &dc)
{
    if (dc.IsClang())
        return llvm::dyn_cast<clang::CXXMethodDecl>((clang::DeclContext *)dc.GetOpaqueDeclContext());
    return nullptr;
}

clang::FunctionDecl *
ClangASTContext::DeclContextGetAsFunctionDecl (const CompilerDeclContext &dc)
{
    if (dc.IsClang())
        return llvm::dyn_cast<clang::FunctionDecl>((clang::DeclContext *)dc.GetOpaqueDeclContext());
    return nullptr;
}

clang::NamespaceDecl *
ClangASTContext::DeclContextGetAsNamespaceDecl (const CompilerDeclContext &dc)
{
    if (dc.IsClang())
        return llvm::dyn_cast<clang::NamespaceDecl>((clang::DeclContext *)dc.GetOpaqueDeclContext());
    return nullptr;
}

ClangASTMetadata *
ClangASTContext::DeclContextGetMetaData (const CompilerDeclContext &dc, const void *object)
{
    clang::ASTContext *ast = DeclContextGetClangASTContext (dc);
    if (ast)
        return ClangASTContext::GetMetadata (ast, object);
    return nullptr;
}

clang::ASTContext *
ClangASTContext::DeclContextGetClangASTContext (const CompilerDeclContext &dc)
{
    ClangASTContext *ast = llvm::dyn_cast_or_null<ClangASTContext>(dc.GetTypeSystem());
    if (ast)
        return ast->getASTContext();
    return nullptr;
}

ClangASTContextForExpressions::ClangASTContextForExpressions (Target &target) :
    ClangASTContext (target.GetArchitecture().GetTriple().getTriple().c_str()),
    m_target_wp(target.shared_from_this()),
    m_persistent_variables (new ClangPersistentVariables)
{
}

UserExpression *
ClangASTContextForExpressions::GetUserExpression (const char *expr,
                   const char *expr_prefix,
                   lldb::LanguageType language,
                   Expression::ResultType desired_type,
                   const EvaluateExpressionOptions &options)
{
    TargetSP target_sp = m_target_wp.lock();
    if (!target_sp)
        return nullptr;
    
    return new ClangUserExpression(*target_sp.get(), expr, expr_prefix, language, desired_type, options);
}

FunctionCaller *
ClangASTContextForExpressions::GetFunctionCaller (const CompilerType &return_type,
                                                  const Address& function_address,
                                                  const ValueList &arg_value_list,
                                                  const char *name)
{
    TargetSP target_sp = m_target_wp.lock();
    if (!target_sp)
        return nullptr;
    
   Process *process = target_sp->GetProcessSP().get();
   if (!process)
       return nullptr;
   
   return new ClangFunctionCaller (*process, return_type, function_address, arg_value_list, name);
}

UtilityFunction *
ClangASTContextForExpressions::GetUtilityFunction (const char *text,
                                                   const char *name)
{
    TargetSP target_sp = m_target_wp.lock();
    if (!target_sp)
        return nullptr;
    
    return new ClangUtilityFunction(*target_sp.get(), text, name);
}

PersistentExpressionState *
ClangASTContextForExpressions::GetPersistentExpressionState ()
{
    return m_persistent_variables.get();
}
