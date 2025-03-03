//===- unittests/CodeGen/DroppedVariableStatsMIRTest.cpp ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/DroppedVariableStatsMIR.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/CodeGen/MIRParser/MIRParser.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Pass.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "gtest/gtest.h"
#include <gtest/gtest.h>
#include <llvm/ADT/SmallString.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassInstrumentation.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/PassTimingInfo.h>
#include <llvm/Support/raw_ostream.h>

using namespace llvm;

namespace {

std::unique_ptr<TargetMachine>
createTargetMachine(std::string TT, StringRef CPU, StringRef FS) {
  std::string Error;
  const Target *T = TargetRegistry::lookupTarget(TT, Error);
  if (!T)
    return nullptr;
  TargetOptions Options;
  return std::unique_ptr<TargetMachine>(
      static_cast<TargetMachine *>(T->createTargetMachine(
          TT, CPU, FS, Options, std::nullopt, std::nullopt)));
}

std::unique_ptr<Module> parseMIR(const TargetMachine &TM, StringRef MIRCode,
                                 MachineModuleInfo &MMI, LLVMContext *Context) {
  SMDiagnostic Diagnostic;
  std::unique_ptr<Module> M;
  std::unique_ptr<MemoryBuffer> MBuffer = MemoryBuffer::getMemBuffer(MIRCode);
  auto MIR = createMIRParser(std::move(MBuffer), *Context);
  if (!MIR)
    return nullptr;

  std::unique_ptr<Module> Mod = MIR->parseIRModule();
  if (!Mod)
    return nullptr;

  Mod->setDataLayout(TM.createDataLayout());

  if (MIR->parseMachineFunctions(*Mod, MMI)) {
    M.reset();
    return nullptr;
  }
  return Mod;
}
// This test ensures that if a DBG_VALUE and an instruction that exists in the
// same scope as that DBG_VALUE are both deleted as a result of an optimization
// pass, debug information is considered not dropped.
TEST(DroppedVariableStatsMIR, BothDeleted) {
  InitializeAllTargetInfos();
  InitializeAllTargets();
  InitializeAllTargetMCs();
  PassInstrumentationCallbacks PIC;
  PassInstrumentation PI(&PIC);

  LLVMContext C;

  const char *MIR =
      R"(
--- |
  ; ModuleID = '/tmp/test.ll'
  source_filename = "/tmp/test.ll"
  target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:64-S128-Fn32"
  
  define noundef range(i32 -2147483647, -2147483648) i32 @_Z3fooi(i32 noundef %x) local_unnamed_addr !dbg !4 {
  entry:
      #dbg_value(i32 %x, !10, !DIExpression(), !11)
    %add = add nsw i32 %x, 1, !dbg !12
    ret i32 0
  }
  
  !llvm.dbg.cu = !{!0}
  !llvm.module.flags = !{!2}
  !llvm.ident = !{!3}
  
  !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: Apple, sysroot: "/")
  !1 = !DIFile(filename: "/tmp/code.cpp", directory: "/")
  !2 = !{i32 2, !"Debug Info Version", i32 3}
  !3 = !{!"clang"}
  !4 = distinct !DISubprogram(name: "foo", linkageName: "_Z3fooi", scope: !5, file: !5, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !9)
  !5 = !DIFile(filename: "/tmp/code.cpp", directory: "")
  !6 = !DISubroutineType(types: !7)
  !7 = !{!8, !8}
  !8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
  !9 = !{!10}
  !10 = !DILocalVariable(name: "x", arg: 1, scope: !4, file: !5, line: 1, type: !8)
  !11 = !DILocation(line: 0, scope: !4)
  !12 = !DILocation(line: 2, column: 11, scope: !4)

...
---
name:            _Z3fooi
alignment:       4
exposesReturnsTwice: false
legalized:       false
regBankSelected: false
selected:        false
failedISel:      false
tracksRegLiveness: true
hasWinCFI:       false
noPhis:          false
isSSA:           true
noVRegs:         false
hasFakeUses:     false
callsEHReturn:   false
callsUnwindInit: false
hasEHCatchret:   false
hasEHScopes:     false
hasEHFunclets:   false
isOutlined:      false
debugInstrRef:   false
failsVerification: false
tracksDebugUserValues: false
registers:
  - { id: 0, class: _, preferred-register: '', flags: [  ] }
  - { id: 1, class: _, preferred-register: '', flags: [  ] }
  - { id: 2, class: _, preferred-register: '', flags: [  ] }
  - { id: 3, class: _, preferred-register: '', flags: [  ] }
liveins:
  - { reg: '$w0', virtual-reg: '' }
frameInfo:
  isFrameAddressTaken: false
  isReturnAddressTaken: false
  hasStackMap:     false
  hasPatchPoint:   false
  stackSize:       0
  offsetAdjustment: 0
  maxAlignment:    1
  adjustsStack:    false
  hasCalls:        false
  stackProtector:  ''
  functionContext: ''
  maxCallFrameSize: 4294967295
  cvBytesOfCalleeSavedRegisters: 0
  hasOpaqueSPAdjustment: false
  hasVAStart:      false
  hasMustTailInVarArgFunc: false
  hasTailCall:     false
  isCalleeSavedInfoValid: false
  localFrameSize:  0
  savePoint:       ''
  restorePoint:    ''
fixedStack:      []
stack:           []
entry_values:    []
callSites:       []
debugValueSubstitutions: []
constants:       []
machineFunctionInfo: {}
body:             |
  bb.1.entry:
    liveins: $w0
  
    %0:_(s32) = COPY $w0
    %1:_(s32) = G_CONSTANT i32 1
    %3:_(s32) = G_CONSTANT i32 0
    DBG_VALUE %0(s32), $noreg, !10, !DIExpression(), debug-location !11
    %2:_(s32) = nsw G_ADD %0, %1, debug-location !12
    $w0 = COPY %3(s32)
    RET_ReallyLR implicit $w0
    )";
  auto TM = createTargetMachine(Triple::normalize("aarch64--"), "", "");
  if (!TM)
    return;
  MachineModuleInfo MMI(TM.get());
  std::unique_ptr<Module> M = parseMIR(*TM, MIR, MMI, &C);
  ASSERT_TRUE(M);

  DroppedVariableStatsMIR Stats;
  auto *MF = MMI.getMachineFunction(*M->getFunction("_Z3fooi"));
  Stats.runBeforePass("Test", MF);

  // This loop simulates an IR pass that drops debug information.
  for (auto &MBB : *MF) {
    for (auto &MI : MBB) {
      if (MI.isDebugValueLike()) {
        MI.eraseFromParent();
        break;
      }
    }
    for (auto &MI : MBB) {
      auto *DbgLoc = MI.getDebugLoc().get();
      if (DbgLoc) {
        MI.eraseFromParent();
        break;
      }
    }
    break;
  }

  Stats.runAfterPass("Test", MF);
  ASSERT_EQ(Stats.getPassDroppedVariables(), false);
}

// This test ensures that if a DBG_VALUE is dropped after an optimization pass,
// but an instruction that shares the same scope as the DBG_VALUE still exists,
// debug information is conisdered dropped.
TEST(DroppedVariableStatsMIR, DbgValLost) {
  InitializeAllTargetInfos();
  InitializeAllTargets();
  InitializeAllTargetMCs();
  PassInstrumentationCallbacks PIC;
  PassInstrumentation PI(&PIC);

  LLVMContext C;

  const char *MIR =
      R"(
--- |
  ; ModuleID = '/tmp/test.ll'
  source_filename = "/tmp/test.ll"
  target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:64-S128-Fn32"
  
  define noundef range(i32 -2147483647, -2147483648) i32 @_Z3fooi(i32 noundef %x) local_unnamed_addr !dbg !4 {
  entry:
      #dbg_value(i32 %x, !10, !DIExpression(), !11)
    %add = add nsw i32 %x, 1, !dbg !12
    ret i32 0
  }
  
  !llvm.dbg.cu = !{!0}
  !llvm.module.flags = !{!2}
  !llvm.ident = !{!3}
  
  !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: Apple, sysroot: "/")
  !1 = !DIFile(filename: "/tmp/code.cpp", directory: "/")
  !2 = !{i32 2, !"Debug Info Version", i32 3}
  !3 = !{!"clang"}
  !4 = distinct !DISubprogram(name: "foo", linkageName: "_Z3fooi", scope: !5, file: !5, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !9)
  !5 = !DIFile(filename: "/tmp/code.cpp", directory: "")
  !6 = !DISubroutineType(types: !7)
  !7 = !{!8, !8}
  !8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
  !9 = !{!10}
  !10 = !DILocalVariable(name: "x", arg: 1, scope: !4, file: !5, line: 1, type: !8)
  !11 = !DILocation(line: 0, scope: !4)
  !12 = !DILocation(line: 2, column: 11, scope: !4)

...
---
name:            _Z3fooi
alignment:       4
exposesReturnsTwice: false
legalized:       false
regBankSelected: false
selected:        false
failedISel:      false
tracksRegLiveness: true
hasWinCFI:       false
noPhis:          false
isSSA:           true
noVRegs:         false
hasFakeUses:     false
callsEHReturn:   false
callsUnwindInit: false
hasEHCatchret:   false
hasEHScopes:     false
hasEHFunclets:   false
isOutlined:      false
debugInstrRef:   false
failsVerification: false
tracksDebugUserValues: false
registers:
  - { id: 0, class: _, preferred-register: '', flags: [  ] }
  - { id: 1, class: _, preferred-register: '', flags: [  ] }
  - { id: 2, class: _, preferred-register: '', flags: [  ] }
  - { id: 3, class: _, preferred-register: '', flags: [  ] }
liveins:
  - { reg: '$w0', virtual-reg: '' }
frameInfo:
  isFrameAddressTaken: false
  isReturnAddressTaken: false
  hasStackMap:     false
  hasPatchPoint:   false
  stackSize:       0
  offsetAdjustment: 0
  maxAlignment:    1
  adjustsStack:    false
  hasCalls:        false
  stackProtector:  ''
  functionContext: ''
  maxCallFrameSize: 4294967295
  cvBytesOfCalleeSavedRegisters: 0
  hasOpaqueSPAdjustment: false
  hasVAStart:      false
  hasMustTailInVarArgFunc: false
  hasTailCall:     false
  isCalleeSavedInfoValid: false
  localFrameSize:  0
  savePoint:       ''
  restorePoint:    ''
fixedStack:      []
stack:           []
entry_values:    []
callSites:       []
debugValueSubstitutions: []
constants:       []
machineFunctionInfo: {}
body:             |
  bb.1.entry:
    liveins: $w0
  
    %0:_(s32) = COPY $w0
    %1:_(s32) = G_CONSTANT i32 1
    %3:_(s32) = G_CONSTANT i32 0
    DBG_VALUE %0(s32), $noreg, !10, !DIExpression(), debug-location !11
    %2:_(s32) = nsw G_ADD %0, %1, debug-location !12
    $w0 = COPY %3(s32)
    RET_ReallyLR implicit $w0
    )";
  auto TM = createTargetMachine(Triple::normalize("aarch64--"), "", "");
  if (!TM)
    return;
  MachineModuleInfo MMI(TM.get());
  std::unique_ptr<Module> M = parseMIR(*TM, MIR, MMI, &C);
  ASSERT_TRUE(M);

  DroppedVariableStatsMIR Stats;
  auto *MF = MMI.getMachineFunction(*M->getFunction("_Z3fooi"));
  Stats.runBeforePass("Test", MF);

  // This loop simulates an IR pass that drops debug information.
  for (auto &MBB : *MF) {
    for (auto &MI : MBB) {
      if (MI.isDebugValueLike()) {
        MI.eraseFromParent();
        break;
      }
    }
    break;
  }

  Stats.runAfterPass("Test", MF);
  ASSERT_EQ(Stats.getPassDroppedVariables(), true);
}

// This test ensures that if a #dbg_value is dropped after an optimization pass,
// but an instruction that has an unrelated scope as the #dbg_value still
// exists, debug information is conisdered not dropped.
TEST(DroppedVariableStatsMIR, UnrelatedScopes) {
  InitializeAllTargetInfos();
  InitializeAllTargets();
  InitializeAllTargetMCs();
  PassInstrumentationCallbacks PIC;
  PassInstrumentation PI(&PIC);

  LLVMContext C;

  const char *MIR =
      R"(
--- |
  ; ModuleID = '/tmp/test.ll'
  source_filename = "/tmp/test.ll"
  target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:64-S128-Fn32"
  
  define noundef range(i32 -2147483647, -2147483648) i32 @_Z3fooi(i32 noundef %x) local_unnamed_addr !dbg !4 {
  entry:
      #dbg_value(i32 %x, !10, !DIExpression(), !11)
    %add = add nsw i32 %x, 1, !dbg !12
    ret i32 0
  }
  
  !llvm.dbg.cu = !{!0}
  !llvm.module.flags = !{!2}
  !llvm.ident = !{!3}
  
  !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: Apple, sysroot: "/")
  !1 = !DIFile(filename: "/tmp/code.cpp", directory: "/")
  !2 = !{i32 2, !"Debug Info Version", i32 3}
  !3 = !{!"clang"}
  !4 = distinct !DISubprogram(name: "foo", linkageName: "_Z3fooi", scope: !5, file: !5, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !9)
  !5 = !DIFile(filename: "/tmp/code.cpp", directory: "")
  !6 = !DISubroutineType(types: !7)
  !7 = !{!8, !8}
  !8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
  !9 = !{!10}
  !10 = !DILocalVariable(name: "x", arg: 1, scope: !4, file: !5, line: 1, type: !8)
  !11 = !DILocation(line: 0, scope: !4)
  !12 = !DILocation(line: 2, column: 11, scope: !13)
  !13 = distinct !DISubprogram(name: "bar", linkageName: "_Z3bari", scope: !5, file: !5, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !9)

...
---
name:            _Z3fooi
alignment:       4
exposesReturnsTwice: false
legalized:       false
regBankSelected: false
selected:        false
failedISel:      false
tracksRegLiveness: true
hasWinCFI:       false
noPhis:          false
isSSA:           true
noVRegs:         false
hasFakeUses:     false
callsEHReturn:   false
callsUnwindInit: false
hasEHCatchret:   false
hasEHScopes:     false
hasEHFunclets:   false
isOutlined:      false
debugInstrRef:   false
failsVerification: false
tracksDebugUserValues: false
registers:
  - { id: 0, class: _, preferred-register: '', flags: [  ] }
  - { id: 1, class: _, preferred-register: '', flags: [  ] }
  - { id: 2, class: _, preferred-register: '', flags: [  ] }
  - { id: 3, class: _, preferred-register: '', flags: [  ] }
liveins:
  - { reg: '$w0', virtual-reg: '' }
frameInfo:
  isFrameAddressTaken: false
  isReturnAddressTaken: false
  hasStackMap:     false
  hasPatchPoint:   false
  stackSize:       0
  offsetAdjustment: 0
  maxAlignment:    1
  adjustsStack:    false
  hasCalls:        false
  stackProtector:  ''
  functionContext: ''
  maxCallFrameSize: 4294967295
  cvBytesOfCalleeSavedRegisters: 0
  hasOpaqueSPAdjustment: false
  hasVAStart:      false
  hasMustTailInVarArgFunc: false
  hasTailCall:     false
  isCalleeSavedInfoValid: false
  localFrameSize:  0
  savePoint:       ''
  restorePoint:    ''
fixedStack:      []
stack:           []
entry_values:    []
callSites:       []
debugValueSubstitutions: []
constants:       []
machineFunctionInfo: {}
body:             |
  bb.1.entry:
    liveins: $w0
  
    %0:_(s32) = COPY $w0
    %1:_(s32) = G_CONSTANT i32 1
    %3:_(s32) = G_CONSTANT i32 0
    DBG_VALUE %0(s32), $noreg, !10, !DIExpression(), debug-location !11
    %2:_(s32) = nsw G_ADD %0, %1, debug-location !12
    $w0 = COPY %3(s32)
    RET_ReallyLR implicit $w0
    )";
  auto TM = createTargetMachine(Triple::normalize("aarch64--"), "", "");
  if (!TM)
    return;
  MachineModuleInfo MMI(TM.get());
  std::unique_ptr<Module> M = parseMIR(*TM, MIR, MMI, &C);
  ASSERT_TRUE(M);

  DroppedVariableStatsMIR Stats;
  auto *MF = MMI.getMachineFunction(*M->getFunction("_Z3fooi"));
  Stats.runBeforePass("Test", MF);

  // This loop simulates an IR pass that drops debug information.
  for (auto &MBB : *MF) {
    for (auto &MI : MBB) {
      if (MI.isDebugValueLike()) {
        MI.eraseFromParent();
        break;
      }
    }
    break;
  }

  Stats.runAfterPass("Test", MF);
  ASSERT_EQ(Stats.getPassDroppedVariables(), false);
}

// This test ensures that if a #dbg_value is dropped after an optimization pass,
// but an instruction that has a scope which is a child of the #dbg_value scope
// still exists, debug information is conisdered dropped.
TEST(DroppedVariableStatsMIR, ChildScopes) {
  InitializeAllTargetInfos();
  InitializeAllTargets();
  InitializeAllTargetMCs();
  PassInstrumentationCallbacks PIC;
  PassInstrumentation PI(&PIC);

  LLVMContext C;

  const char *MIR =
      R"(
--- |
  ; ModuleID = '/tmp/test.ll'
  source_filename = "/tmp/test.ll"
  target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:64-S128-Fn32"
  
  define noundef range(i32 -2147483647, -2147483648) i32 @_Z3fooi(i32 noundef %x) local_unnamed_addr !dbg !4 {
  entry:
      #dbg_value(i32 %x, !10, !DIExpression(), !11)
    %add = add nsw i32 %x, 1, !dbg !12
    ret i32 0
  }
  
  !llvm.dbg.cu = !{!0}
  !llvm.module.flags = !{!2}
  !llvm.ident = !{!3}
  
  !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: Apple, sysroot: "/")
  !1 = !DIFile(filename: "/tmp/code.cpp", directory: "/")
  !2 = !{i32 2, !"Debug Info Version", i32 3}
  !3 = !{!"clang"}
  !4 = distinct !DISubprogram(name: "foo", linkageName: "_Z3fooi", scope: !5, file: !5, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !9)
  !5 = !DIFile(filename: "/tmp/code.cpp", directory: "")
  !6 = !DISubroutineType(types: !7)
  !7 = !{!8, !8}
  !8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
  !9 = !{!10}
  !10 = !DILocalVariable(name: "x", arg: 1, scope: !4, file: !5, line: 1, type: !8)
  !11 = !DILocation(line: 0, scope: !4)
  !12 = !DILocation(line: 2, column: 11, scope: !13)
  !13 = distinct !DILexicalBlock(scope: !4, file: !5, line: 10, column: 28)

...
---
name:            _Z3fooi
alignment:       4
exposesReturnsTwice: false
legalized:       false
regBankSelected: false
selected:        false
failedISel:      false
tracksRegLiveness: true
hasWinCFI:       false
noPhis:          false
isSSA:           true
noVRegs:         false
hasFakeUses:     false
callsEHReturn:   false
callsUnwindInit: false
hasEHCatchret:   false
hasEHScopes:     false
hasEHFunclets:   false
isOutlined:      false
debugInstrRef:   false
failsVerification: false
tracksDebugUserValues: false
registers:
  - { id: 0, class: _, preferred-register: '', flags: [  ] }
  - { id: 1, class: _, preferred-register: '', flags: [  ] }
  - { id: 2, class: _, preferred-register: '', flags: [  ] }
  - { id: 3, class: _, preferred-register: '', flags: [  ] }
liveins:
  - { reg: '$w0', virtual-reg: '' }
frameInfo:
  isFrameAddressTaken: false
  isReturnAddressTaken: false
  hasStackMap:     false
  hasPatchPoint:   false
  stackSize:       0
  offsetAdjustment: 0
  maxAlignment:    1
  adjustsStack:    false
  hasCalls:        false
  stackProtector:  ''
  functionContext: ''
  maxCallFrameSize: 4294967295
  cvBytesOfCalleeSavedRegisters: 0
  hasOpaqueSPAdjustment: false
  hasVAStart:      false
  hasMustTailInVarArgFunc: false
  hasTailCall:     false
  isCalleeSavedInfoValid: false
  localFrameSize:  0
  savePoint:       ''
  restorePoint:    ''
fixedStack:      []
stack:           []
entry_values:    []
callSites:       []
debugValueSubstitutions: []
constants:       []
machineFunctionInfo: {}
body:             |
  bb.1.entry:
    liveins: $w0
  
    %0:_(s32) = COPY $w0
    %1:_(s32) = G_CONSTANT i32 1
    %3:_(s32) = G_CONSTANT i32 0
    DBG_VALUE %0(s32), $noreg, !10, !DIExpression(), debug-location !11
    %2:_(s32) = nsw G_ADD %0, %1, debug-location !12
    $w0 = COPY %3(s32)
    RET_ReallyLR implicit $w0
    )";
  auto TM = createTargetMachine(Triple::normalize("aarch64--"), "", "");
  if (!TM)
    return;
  MachineModuleInfo MMI(TM.get());
  std::unique_ptr<Module> M = parseMIR(*TM, MIR, MMI, &C);
  ASSERT_TRUE(M);

  DroppedVariableStatsMIR Stats;
  auto *MF = MMI.getMachineFunction(*M->getFunction("_Z3fooi"));
  Stats.runBeforePass("Test", MF);

  // This loop simulates an IR pass that drops debug information.
  for (auto &MBB : *MF) {
    for (auto &MI : MBB) {
      if (MI.isDebugValueLike()) {
        MI.eraseFromParent();
        break;
      }
    }
    break;
  }

  Stats.runAfterPass("Test", MF);
  ASSERT_EQ(Stats.getPassDroppedVariables(), true);
}

// This test ensures that if a DBG_VALUE is dropped after an optimization pass,
// but an instruction that has a scope which is a child of the DBG_VALUE scope
// still exists, and the DBG_VALUE is inlined at another location, debug
// information is conisdered not dropped.
TEST(DroppedVariableStatsMIR, InlinedAt) {
  InitializeAllTargetInfos();
  InitializeAllTargets();
  InitializeAllTargetMCs();
  PassInstrumentationCallbacks PIC;
  PassInstrumentation PI(&PIC);

  LLVMContext C;

  const char *MIR =
      R"(
--- |
  ; ModuleID = '/tmp/test.ll'
  source_filename = "/tmp/test.ll"
  target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:64-S128-Fn32"
  
  define noundef range(i32 -2147483647, -2147483648) i32 @_Z3fooi(i32 noundef %x) local_unnamed_addr !dbg !4 {
  entry:
      #dbg_value(i32 %x, !10, !DIExpression(), !11)
    %add = add nsw i32 %x, 1, !dbg !12
    ret i32 0
  }
  
  !llvm.dbg.cu = !{!0}
  !llvm.module.flags = !{!2}
  !llvm.ident = !{!3}
  
  !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: Apple, sysroot: "/")
  !1 = !DIFile(filename: "/tmp/code.cpp", directory: "/")
  !2 = !{i32 2, !"Debug Info Version", i32 3}
  !3 = !{!"clang"}
  !4 = distinct !DISubprogram(name: "foo", linkageName: "_Z3fooi", scope: !5, file: !5, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !9)
  !5 = !DIFile(filename: "/tmp/code.cpp", directory: "")
  !6 = !DISubroutineType(types: !7)
  !7 = !{!8, !8}
  !8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
  !9 = !{!10}
  !10 = !DILocalVariable(name: "x", arg: 1, scope: !4, file: !5, line: 1, type: !8)
  !11 = !DILocation(line: 0, scope: !4, inlinedAt: !14)
  !12 = !DILocation(line: 2, column: 11, scope: !13)
  !13 = distinct !DILexicalBlock(scope: !4, file: !5, line: 10, column: 28)
  !14 = !DILocation(line: 3, column: 2, scope: !4)

...
---
name:            _Z3fooi
alignment:       4
exposesReturnsTwice: false
legalized:       false
regBankSelected: false
selected:        false
failedISel:      false
tracksRegLiveness: true
hasWinCFI:       false
noPhis:          false
isSSA:           true
noVRegs:         false
hasFakeUses:     false
callsEHReturn:   false
callsUnwindInit: false
hasEHCatchret:   false
hasEHScopes:     false
hasEHFunclets:   false
isOutlined:      false
debugInstrRef:   false
failsVerification: false
tracksDebugUserValues: false
registers:
  - { id: 0, class: _, preferred-register: '', flags: [  ] }
  - { id: 1, class: _, preferred-register: '', flags: [  ] }
  - { id: 2, class: _, preferred-register: '', flags: [  ] }
  - { id: 3, class: _, preferred-register: '', flags: [  ] }
liveins:
  - { reg: '$w0', virtual-reg: '' }
frameInfo:
  isFrameAddressTaken: false
  isReturnAddressTaken: false
  hasStackMap:     false
  hasPatchPoint:   false
  stackSize:       0
  offsetAdjustment: 0
  maxAlignment:    1
  adjustsStack:    false
  hasCalls:        false
  stackProtector:  ''
  functionContext: ''
  maxCallFrameSize: 4294967295
  cvBytesOfCalleeSavedRegisters: 0
  hasOpaqueSPAdjustment: false
  hasVAStart:      false
  hasMustTailInVarArgFunc: false
  hasTailCall:     false
  isCalleeSavedInfoValid: false
  localFrameSize:  0
  savePoint:       ''
  restorePoint:    ''
fixedStack:      []
stack:           []
entry_values:    []
callSites:       []
debugValueSubstitutions: []
constants:       []
machineFunctionInfo: {}
body:             |
  bb.1.entry:
    liveins: $w0
  
    %0:_(s32) = COPY $w0
    %1:_(s32) = G_CONSTANT i32 1
    %3:_(s32) = G_CONSTANT i32 0
    DBG_VALUE %0(s32), $noreg, !10, !DIExpression(), debug-location !11
    %2:_(s32) = nsw G_ADD %0, %1, debug-location !12
    $w0 = COPY %3(s32)
    RET_ReallyLR implicit $w0
    )";
  auto TM = createTargetMachine(Triple::normalize("aarch64--"), "", "");
  if (!TM)
    return;
  MachineModuleInfo MMI(TM.get());
  std::unique_ptr<Module> M = parseMIR(*TM, MIR, MMI, &C);
  ASSERT_TRUE(M);

  DroppedVariableStatsMIR Stats;
  auto *MF = MMI.getMachineFunction(*M->getFunction("_Z3fooi"));
  Stats.runBeforePass("Test", MF);

  // This loop simulates an IR pass that drops debug information.
  for (auto &MBB : *MF) {
    for (auto &MI : MBB) {
      if (MI.isDebugValueLike()) {
        MI.eraseFromParent();
        break;
      }
    }
    break;
  }

  Stats.runAfterPass("Test", MF);
  ASSERT_EQ(Stats.getPassDroppedVariables(), false);
}

// This test ensures that if a DBG_VALUE is dropped after an optimization pass,
// but an instruction that has a scope which is a child of the DBG_VALUE scope
// still exists, and the DBG_VALUE and the instruction are inlined at another
// location, debug information is conisdered dropped.
TEST(DroppedVariableStatsMIR, InlinedAtShared) {
  InitializeAllTargetInfos();
  InitializeAllTargets();
  InitializeAllTargetMCs();
  PassInstrumentationCallbacks PIC;
  PassInstrumentation PI(&PIC);

  LLVMContext C;

  const char *MIR =
      R"(
--- |
  ; ModuleID = '/tmp/test.ll'
  source_filename = "/tmp/test.ll"
  target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:64-S128-Fn32"
  
  define noundef range(i32 -2147483647, -2147483648) i32 @_Z3fooi(i32 noundef %x) local_unnamed_addr !dbg !4 {
  entry:
      #dbg_value(i32 %x, !10, !DIExpression(), !11)
    %add = add nsw i32 %x, 1, !dbg !12
    ret i32 0
  }
  
  !llvm.dbg.cu = !{!0}
  !llvm.module.flags = !{!2}
  !llvm.ident = !{!3}
  
  !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: Apple, sysroot: "/")
  !1 = !DIFile(filename: "/tmp/code.cpp", directory: "/")
  !2 = !{i32 2, !"Debug Info Version", i32 3}
  !3 = !{!"clang"}
  !4 = distinct !DISubprogram(name: "foo", linkageName: "_Z3fooi", scope: !5, file: !5, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !9)
  !5 = !DIFile(filename: "/tmp/code.cpp", directory: "")
  !6 = !DISubroutineType(types: !7)
  !7 = !{!8, !8}
  !8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
  !9 = !{!10}
  !10 = !DILocalVariable(name: "x", arg: 1, scope: !4, file: !5, line: 1, type: !8)
  !11 = !DILocation(line: 0, scope: !4, inlinedAt: !14)
  !12 = !DILocation(line: 2, column: 11, scope: !13, inlinedAt: !14)
  !13 = distinct !DILexicalBlock(scope: !4, file: !5, line: 10, column: 28)
  !14 = !DILocation(line: 3, column: 2, scope: !4)

...
---
name:            _Z3fooi
alignment:       4
exposesReturnsTwice: false
legalized:       false
regBankSelected: false
selected:        false
failedISel:      false
tracksRegLiveness: true
hasWinCFI:       false
noPhis:          false
isSSA:           true
noVRegs:         false
hasFakeUses:     false
callsEHReturn:   false
callsUnwindInit: false
hasEHCatchret:   false
hasEHScopes:     false
hasEHFunclets:   false
isOutlined:      false
debugInstrRef:   false
failsVerification: false
tracksDebugUserValues: false
registers:
  - { id: 0, class: _, preferred-register: '', flags: [  ] }
  - { id: 1, class: _, preferred-register: '', flags: [  ] }
  - { id: 2, class: _, preferred-register: '', flags: [  ] }
  - { id: 3, class: _, preferred-register: '', flags: [  ] }
liveins:
  - { reg: '$w0', virtual-reg: '' }
frameInfo:
  isFrameAddressTaken: false
  isReturnAddressTaken: false
  hasStackMap:     false
  hasPatchPoint:   false
  stackSize:       0
  offsetAdjustment: 0
  maxAlignment:    1
  adjustsStack:    false
  hasCalls:        false
  stackProtector:  ''
  functionContext: ''
  maxCallFrameSize: 4294967295
  cvBytesOfCalleeSavedRegisters: 0
  hasOpaqueSPAdjustment: false
  hasVAStart:      false
  hasMustTailInVarArgFunc: false
  hasTailCall:     false
  isCalleeSavedInfoValid: false
  localFrameSize:  0
  savePoint:       ''
  restorePoint:    ''
fixedStack:      []
stack:           []
entry_values:    []
callSites:       []
debugValueSubstitutions: []
constants:       []
machineFunctionInfo: {}
body:             |
  bb.1.entry:
    liveins: $w0
  
    %0:_(s32) = COPY $w0
    %1:_(s32) = G_CONSTANT i32 1
    %3:_(s32) = G_CONSTANT i32 0
    DBG_VALUE %0(s32), $noreg, !10, !DIExpression(), debug-location !11
    %2:_(s32) = nsw G_ADD %0, %1, debug-location !12
    $w0 = COPY %3(s32)
    RET_ReallyLR implicit $w0
    )";
  auto TM = createTargetMachine(Triple::normalize("aarch64--"), "", "");
  if (!TM)
    return;
  MachineModuleInfo MMI(TM.get());
  std::unique_ptr<Module> M = parseMIR(*TM, MIR, MMI, &C);
  ASSERT_TRUE(M);

  DroppedVariableStatsMIR Stats;
  auto *MF = MMI.getMachineFunction(*M->getFunction("_Z3fooi"));
  Stats.runBeforePass("Test", MF);

  // This loop simulates an IR pass that drops debug information.
  for (auto &MBB : *MF) {
    for (auto &MI : MBB) {
      if (MI.isDebugValueLike()) {
        MI.eraseFromParent();
        break;
      }
    }
    break;
  }

  Stats.runAfterPass("Test", MF);
  ASSERT_EQ(Stats.getPassDroppedVariables(), true);
}

// This test ensures that if a DBG_VALUE is dropped after an optimization pass,
// but an instruction that has a scope which is a child of the DBG_VALUE scope
// still exists, and the instruction is inlined at a location that is the
// DBG_VALUE's inlined at location, debug information is conisdered dropped.
TEST(DroppedVariableStatsMIR, InlinedAtChild) {
  InitializeAllTargetInfos();
  InitializeAllTargets();
  InitializeAllTargetMCs();
  PassInstrumentationCallbacks PIC;
  PassInstrumentation PI(&PIC);

  LLVMContext C;

  const char *MIR =
      R"(
--- |
  ; ModuleID = '/tmp/test.ll'
  source_filename = "/tmp/test.ll"
  target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:64-S128-Fn32"
  
  define noundef range(i32 -2147483647, -2147483648) i32 @_Z3fooi(i32 noundef %x) local_unnamed_addr !dbg !4 {
  entry:
      #dbg_value(i32 %x, !10, !DIExpression(), !11)
    %add = add nsw i32 %x, 1, !dbg !12
    ret i32 0
  }
  
  !llvm.dbg.cu = !{!0}
  !llvm.module.flags = !{!2}
  !llvm.ident = !{!3}
  
  !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: Apple, sysroot: "/")
  !1 = !DIFile(filename: "/tmp/code.cpp", directory: "/")
  !2 = !{i32 2, !"Debug Info Version", i32 3}
  !3 = !{!"clang"}
  !4 = distinct !DISubprogram(name: "foo", linkageName: "_Z3fooi", scope: !5, file: !5, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !9)
  !5 = !DIFile(filename: "/tmp/code.cpp", directory: "")
  !6 = !DISubroutineType(types: !7)
  !7 = !{!8, !8}
  !8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
  !9 = !{!10}
  !10 = !DILocalVariable(name: "x", arg: 1, scope: !4, file: !5, line: 1, type: !8)
  !11 = !DILocation(line: 0, scope: !4, inlinedAt: !14)
  !12 = !DILocation(line: 2, column: 11, scope: !13, inlinedAt: !15)
  !13 = distinct !DILexicalBlock(scope: !4, file: !5, line: 10, column: 28)
  !14 = !DILocation(line: 3, column: 2, scope: !4)
  !15 = !DILocation(line: 4, column: 5, scope: !13, inlinedAt: !14)

...
---
name:            _Z3fooi
alignment:       4
exposesReturnsTwice: false
legalized:       false
regBankSelected: false
selected:        false
failedISel:      false
tracksRegLiveness: true
hasWinCFI:       false
noPhis:          false
isSSA:           true
noVRegs:         false
hasFakeUses:     false
callsEHReturn:   false
callsUnwindInit: false
hasEHCatchret:   false
hasEHScopes:     false
hasEHFunclets:   false
isOutlined:      false
debugInstrRef:   false
failsVerification: false
tracksDebugUserValues: false
registers:
  - { id: 0, class: _, preferred-register: '', flags: [  ] }
  - { id: 1, class: _, preferred-register: '', flags: [  ] }
  - { id: 2, class: _, preferred-register: '', flags: [  ] }
  - { id: 3, class: _, preferred-register: '', flags: [  ] }
liveins:
  - { reg: '$w0', virtual-reg: '' }
frameInfo:
  isFrameAddressTaken: false
  isReturnAddressTaken: false
  hasStackMap:     false
  hasPatchPoint:   false
  stackSize:       0
  offsetAdjustment: 0
  maxAlignment:    1
  adjustsStack:    false
  hasCalls:        false
  stackProtector:  ''
  functionContext: ''
  maxCallFrameSize: 4294967295
  cvBytesOfCalleeSavedRegisters: 0
  hasOpaqueSPAdjustment: false
  hasVAStart:      false
  hasMustTailInVarArgFunc: false
  hasTailCall:     false
  isCalleeSavedInfoValid: false
  localFrameSize:  0
  savePoint:       ''
  restorePoint:    ''
fixedStack:      []
stack:           []
entry_values:    []
callSites:       []
debugValueSubstitutions: []
constants:       []
machineFunctionInfo: {}
body:             |
  bb.1.entry:
    liveins: $w0
  
    %0:_(s32) = COPY $w0
    %1:_(s32) = G_CONSTANT i32 1
    %3:_(s32) = G_CONSTANT i32 0
    DBG_VALUE %0(s32), $noreg, !10, !DIExpression(), debug-location !11
    %2:_(s32) = nsw G_ADD %0, %1, debug-location !12
    $w0 = COPY %3(s32)
    RET_ReallyLR implicit $w0
    )";
  auto TM = createTargetMachine(Triple::normalize("aarch64--"), "", "");
  if (!TM)
    return;
  MachineModuleInfo MMI(TM.get());
  std::unique_ptr<Module> M = parseMIR(*TM, MIR, MMI, &C);
  ASSERT_TRUE(M);

  DroppedVariableStatsMIR Stats;
  auto *MF = MMI.getMachineFunction(*M->getFunction("_Z3fooi"));
  Stats.runBeforePass("Test", MF);

  // This loop simulates an IR pass that drops debug information.
  for (auto &MBB : *MF) {
    for (auto &MI : MBB) {
      if (MI.isDebugValueLike()) {
        MI.eraseFromParent();
        break;
      }
    }
    break;
  }

  Stats.runAfterPass("Test", MF);
  ASSERT_EQ(Stats.getPassDroppedVariables(), true);
}

} // end anonymous namespace
