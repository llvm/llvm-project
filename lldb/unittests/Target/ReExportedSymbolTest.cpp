//===-- ReExportedSymbolTest.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/ObjectFile/ELF/ObjectFileELF.h"
#include "Plugins/Platform/Linux/PlatformLinux.h"
#include "Plugins/SymbolFile/Symtab/SymbolFileSymtab.h"
#include "TestingSupport/SubsystemRAII.h"
#include "TestingSupport/TestUtilities.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/Symbol.h"
#include "lldb/Symbol/Symtab.h"
#include "lldb/Target/Platform.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/ArchSpec.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace lldb_private;
using namespace lldb;

namespace {
class ReExportedSymbolTest : public testing::Test {
  SubsystemRAII<FileSystem, HostInfo, ObjectFileELF, SymbolFileSymtab,
                platform_linux::PlatformLinux>
      subsystems;
};
} // namespace

// An ELF filter library naming two auxiliary filtees.  Its zero-sized
// exported functions "foo", "bar" and "open@@FBSD_1.0" are placeholders the
// dynamic linker resolves through the filtees.  ObjectFileELF leaves them as
// plain code symbols (re-export marking would clobber their address ranges);
// Symbol::ResolveReExportedSymbol shadows them at resolution time.  "bar" is
// defined by the first filtee, "foo" only by the second, so resolving "foo"
// exercises the ordered search across filtees.  "open@@FBSD_1.0" exercises
// version-suffix stripping, and the local "hidden_helper" must not be
// shadowed even though the first filtee exports that name.
//
// .dynstr content: "\0libaux1.so\0libaux2.so\0"
//   offset 0x1: "libaux1.so"
//   offset 0xC: "libaux2.so"
static const char *k_filter_yaml = R"(
--- !ELF
FileHeader:
  Class:           ELFCLASS64
  Data:            ELFDATA2LSB
  Type:            ET_DYN
  Machine:         EM_X86_64
Sections:
  - Name:            .text
    Type:            SHT_PROGBITS
    Flags:           [ SHF_ALLOC, SHF_EXECINSTR ]
    Address:         0x1000
    AddressAlign:    0x10
    Content:         C3C3C3C3C3C3C3C3
  - Name:            .dynstr
    Type:            SHT_STRTAB
    Flags:           [ SHF_ALLOC ]
    Address:         0x2000
    AddressAlign:    0x1
    Content:         "006C6962617578312E736F006C6962617578322E736F00"
  - Name:            .dynamic
    Type:            SHT_DYNAMIC
    Flags:           [ SHF_ALLOC ]
    Address:         0x3000
    AddressAlign:    0x8
    Link:            .dynstr
    Entries:
      - Tag:             DT_AUXILIARY
        Value:           0x1
      - Tag:             DT_AUXILIARY
        Value:           0xC
      - Tag:             DT_NULL
        Value:           0x0
Symbols:
  - Name:            hidden_helper
    Type:            STT_FUNC
    Section:         .text
    Value:           0x1003
    Binding:         STB_LOCAL
  - Name:            foo
    Type:            STT_FUNC
    Section:         .text
    Value:           0x1000
    Binding:         STB_GLOBAL
  - Name:            bar
    Type:            STT_FUNC
    Section:         .text
    Value:           0x1001
    Binding:         STB_GLOBAL
  - Name:            "open@@FBSD_1.0"
    Type:            STT_FUNC
    Section:         .text
    Value:           0x1001
    Binding:         STB_WEAK
  # A real (nonzero-sized) function the filter defines itself.  Besides being
  # realistic -- a filter library like libc.so.7 is full of real functions --
  # it gives the object file a code symbol so that SymbolFileSymtab claims the
  # module (a module whose only symbols are re-export placeholders has no
  # abilities and would get no symbol table).
  - Name:            filter_local
    Type:            STT_FUNC
    Section:         .text
    Value:           0x1002
    Size:            0x2
    Binding:         STB_GLOBAL
  # A genuine (nonzero-sized) implementation that the first filtee *also*
  # defines.  The dynamic linker always searches the filtees before the
  # filter object's own definition, so the filtee must win even though the
  # filter's definition is real code and not a placeholder.
  - Name:            dup_impl
    Type:            STT_FUNC
    Section:         .text
    Value:           0x1004
    Size:            0x2
    Binding:         STB_GLOBAL
  # A placeholder whose name *both* filtees define.  The search stops at the
  # first filtee that provides it, so the second filtee's definition is never
  # reached.
  - Name:            in_both_filtees
    Type:            STT_FUNC
    Section:         .text
    Value:           0x1005
    Binding:         STB_GLOBAL
...
)";

static const char *k_aux1_yaml = R"(
--- !ELF
FileHeader:
  Class:           ELFCLASS64
  Data:            ELFDATA2LSB
  Type:            ET_DYN
  Machine:         EM_X86_64
Sections:
  - Name:            .text
    Type:            SHT_PROGBITS
    Flags:           [ SHF_ALLOC, SHF_EXECINSTR ]
    Address:         0x1000
    AddressAlign:    0x10
    Content:         C3C3C3C3C3C3C3C3
Symbols:
  - Name:            bar
    Type:            STT_FUNC
    Section:         .text
    Value:           0x1000
    Size:            0x2
    Binding:         STB_GLOBAL
  - Name:            hidden_helper
    Type:            STT_FUNC
    Section:         .text
    Value:           0x1002
    Size:            0x2
    Binding:         STB_GLOBAL
  - Name:            dup_impl
    Type:            STT_FUNC
    Section:         .text
    Value:           0x1004
    Size:            0x2
    Binding:         STB_GLOBAL
  - Name:            in_both_filtees
    Type:            STT_FUNC
    Section:         .text
    Value:           0x1006
    Size:            0x2
    Binding:         STB_GLOBAL
...
)";

// The second filtee's definitions are weak: that is how FreeBSD's
// libsys.so.7 exports its syscall stubs, and weak definitions take part in
// dynamic linking just like global ones.
static const char *k_aux2_yaml = R"(
--- !ELF
FileHeader:
  Class:           ELFCLASS64
  Data:            ELFDATA2LSB
  Type:            ET_DYN
  Machine:         EM_X86_64
Sections:
  - Name:            .text
    Type:            SHT_PROGBITS
    Flags:           [ SHF_ALLOC, SHF_EXECINSTR ]
    Address:         0x1000
    AddressAlign:    0x10
    Content:         C3C3C3C3C3C3
Symbols:
  - Name:            foo
    Type:            STT_FUNC
    Section:         .text
    Value:           0x1000
    Size:            0x2
    Binding:         STB_WEAK
  - Name:            open
    Type:            STT_FUNC
    Section:         .text
    Value:           0x1002
    Size:            0x2
    Binding:         STB_WEAK
  - Name:            in_both_filtees
    Type:            STT_FUNC
    Section:         .text
    Value:           0x1004
    Size:            0x2
    Binding:         STB_WEAK
...
)";

// Creates a module from \p file's in-memory buffer, gives it the name the
// filter's dynamic section refers to, and appends it to the target's image
// list.
static ModuleSP AddModule(Target &target, TestFile &file, const char *name) {
  ModuleSP module_sp = std::make_shared<Module>(file.moduleSpec());

  // The module is built from a buffer with no backing file.  Parse everything
  // that depends on that buffer *now*, while the module still has no file:
  //  - GetReExportedLibraries() parses and caches the dynamic section, which
  //    re-export resolution reads later.
  //  - GetSymtab() parses and caches the symbol table (and its symbol file).
  // Renaming afterwards would otherwise make these look for the symbols in the
  // now-nonexistent file.  The name is required so re-export resolution can
  // find the module in the target by the file spec recorded in the filter.
  if (ObjectFile *of = module_sp->GetObjectFile())
    of->GetReExportedLibraries();
  module_sp->GetSymtab();
  module_sp->SetFileSpecAndObjectName(FileSpec(name), ConstString());

  // Append without notifying: the module-added notification pulls in symbol
  // preloading and scripting-resource loading, which need a fully initialized
  // Debugger.  This test only needs the module to be findable in the target's
  // image list.
  target.GetImages().Append(module_sp, /*notify=*/false);
  return module_sp;
}

// Find the symbol whose name is \p name, ignoring any @VERSION suffix, in
// \p module_sp by iterating the symbol table.  Iterating keeps the lookup
// independent of how the symtab name indexes treat versioned names, which is
// orthogonal to what is tested here.
static const Symbol *FindFilterSymbol(const ModuleSP &module_sp,
                                      llvm::StringRef name) {
  Symtab *symtab = module_sp->GetSymtab();
  if (!symtab)
    return nullptr;
  for (size_t i = 0, e = symtab->GetNumSymbols(); i != e; ++i) {
    const Symbol *symbol = symtab->SymbolAtIndex(i);
    llvm::StringRef symbol_name = symbol->GetName().GetStringRef();
    if (symbol_name.substr(0, symbol_name.find('@')) == name)
      return symbol;
  }
  return nullptr;
}

TEST_F(ReExportedSymbolTest, ResolveThroughFilteesInOrder) {
  // Note: the TestFiles are declared before the target so that their
  // in-memory buffers outlive the modules the target holds.
  auto filter_file = TestFile::fromYaml(k_filter_yaml);
  ASSERT_THAT_EXPECTED(filter_file, llvm::Succeeded());
  auto aux1_file = TestFile::fromYaml(k_aux1_yaml);
  ASSERT_THAT_EXPECTED(aux1_file, llvm::Succeeded());
  auto aux2_file = TestFile::fromYaml(k_aux2_yaml);
  ASSERT_THAT_EXPECTED(aux2_file, llvm::Succeeded());

  ArchSpec arch("x86_64-pc-linux");
  Platform::SetHostPlatform(
      platform_linux::PlatformLinux::CreateInstance(true, &arch));
  DebuggerSP debugger_sp = Debugger::CreateInstance();
  ASSERT_TRUE(debugger_sp);

  PlatformSP platform_sp;
  TargetSP target_sp;
  debugger_sp->GetTargetList().CreateTarget(
      *debugger_sp, "", arch, eLoadDependentsNo, platform_sp, target_sp);
  ASSERT_TRUE(target_sp);

  ModuleSP filter_sp = AddModule(*target_sp, *filter_file, "libfilter.so");
  ASSERT_TRUE(filter_sp);
  ModuleSP aux1_sp = AddModule(*target_sp, *aux1_file, "libaux1.so");
  ASSERT_TRUE(aux1_sp);
  ModuleSP aux2_sp = AddModule(*target_sp, *aux2_file, "libaux2.so");
  ASSERT_TRUE(aux2_sp);

  // Sanity: the filter module parsed its symbols and its DT_AUXILIARY filtees
  // were read in order.  Splitting these out pinpoints which layer fails if
  // resolution below does.
  ObjectFile *filter_of = filter_sp->GetObjectFile();
  ASSERT_NE(nullptr, filter_of);
  FileSpecList filtees = filter_of->GetReExportedLibraries();
  ASSERT_EQ(2u, filtees.GetSize());
  EXPECT_EQ(ConstString("libaux1.so"),
            filtees.GetFileSpecAtIndex(0).GetFilename());
  EXPECT_EQ(ConstString("libaux2.so"),
            filtees.GetFileSpecAtIndex(1).GetFilename());
  Symtab *filter_symtab = filter_sp->GetSymtab();
  ASSERT_NE(nullptr, filter_symtab);
  ASSERT_GT(filter_symtab->GetNumSymbols(), 0u);

  // The placeholders stay plain code symbols: marking them as re-exported
  // would reuse their address ranges as string-pointer storage and corrupt
  // the file-address indexes.
  const Symbol *bar = FindFilterSymbol(filter_sp, "bar");
  ASSERT_NE(nullptr, bar);
  EXPECT_EQ(eSymbolTypeCode, bar->GetType());

  // "bar" is found in the first filtee.
  Symbol *bar_def = bar->ResolveReExportedSymbol(*target_sp, filter_sp);
  ASSERT_NE(nullptr, bar_def);
  EXPECT_EQ(ConstString("bar"), bar_def->GetName());
  EXPECT_EQ(eSymbolTypeCode, bar_def->GetType());
  EXPECT_EQ(aux1_sp, bar_def->GetAddress().GetModule());

  // "foo" is not defined by the first filtee: resolution must continue with
  // the remaining filtees in declaration order and find it in the second.
  const Symbol *foo = FindFilterSymbol(filter_sp, "foo");
  ASSERT_NE(nullptr, foo);
  Symbol *foo_def = foo->ResolveReExportedSymbol(*target_sp, filter_sp);
  ASSERT_NE(nullptr, foo_def);
  EXPECT_EQ(ConstString("foo"), foo_def->GetName());
  EXPECT_EQ(eSymbolTypeCode, foo_def->GetType());
  EXPECT_EQ(aux2_sp, foo_def->GetAddress().GetModule());

  // "open@@FBSD_1.0": the @VERSION suffix is stripped before the filtees are
  // searched, since the filtee exports the plain name.
  const Symbol *open_sym = FindFilterSymbol(filter_sp, "open");
  ASSERT_NE(nullptr, open_sym);
  Symbol *open_def = open_sym->ResolveReExportedSymbol(*target_sp, filter_sp);
  ASSERT_NE(nullptr, open_def);
  EXPECT_EQ(ConstString("open"), open_def->GetName());
  EXPECT_EQ(aux2_sp, open_def->GetAddress().GetModule());

  // A local symbol is invisible to the dynamic linker and must not be
  // shadowed, even though the first filtee exports the same name.
  const Symbol *hidden = FindFilterSymbol(filter_sp, "hidden_helper");
  ASSERT_NE(nullptr, hidden);
  EXPECT_EQ(nullptr, hidden->ResolveReExportedSymbol(*target_sp, filter_sp));

  // A name no filtee provides resolves to nothing; callers then fall back
  // to the filter's own (intact) definition.
  const Symbol *filter_local = FindFilterSymbol(filter_sp, "filter_local");
  ASSERT_NE(nullptr, filter_local);
  EXPECT_EQ(nullptr,
            filter_local->ResolveReExportedSymbol(*target_sp, filter_sp));

  // Both the filter and the first filtee genuinely define "dup_impl"
  // (nonzero-sized code on both sides).  The dynamic linker searches the
  // filtees before the filter object's own definition, so the filtee's
  // definition must win.
  const Symbol *dup = FindFilterSymbol(filter_sp, "dup_impl");
  ASSERT_NE(nullptr, dup);
  EXPECT_EQ(eSymbolTypeCode, dup->GetType());
  Symbol *dup_def = dup->ResolveReExportedSymbol(*target_sp, filter_sp);
  ASSERT_NE(nullptr, dup_def);
  EXPECT_EQ(ConstString("dup_impl"), dup_def->GetName());
  EXPECT_EQ(aux1_sp, dup_def->GetAddress().GetModule());

  // Both filtees define "in_both_filtees".  They are searched in declaration
  // order and the search stops at the first hit, so the definition must come
  // from the first filtee, at its own address, and the second filtee's
  // definition must be ignored.
  const Symbol *in_both = FindFilterSymbol(filter_sp, "in_both_filtees");
  ASSERT_NE(nullptr, in_both);
  Symbol *in_both_def = in_both->ResolveReExportedSymbol(*target_sp, filter_sp);
  ASSERT_NE(nullptr, in_both_def);
  EXPECT_EQ(ConstString("in_both_filtees"), in_both_def->GetName());
  EXPECT_EQ(aux1_sp, in_both_def->GetAddress().GetModule());
  EXPECT_EQ(0x1006u, in_both_def->GetAddress().GetFileAddress());

  // Without the containing module nothing connects the symbol to the
  // filtees, so "foo" cannot be resolved.  This is why callers that have
  // the module should pass it.
  EXPECT_EQ(nullptr, foo->ResolveReExportedSymbol(*target_sp));
}
