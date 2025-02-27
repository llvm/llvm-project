; RUN: opt -passes=debugify -S -o - < %s | FileCheck %s
; RUN: opt -passes=debugify -S -o - < %s | FileCheck %s

; RUN: opt -passes=debugify,debugify -S -o - < %s 2>&1 | \
; RUN:   FileCheck %s -check-prefix=CHECK-REPEAT
; RUN: opt -passes=debugify,debugify -S -o - < %s 2>&1 | \
; RUN:   FileCheck %s -check-prefix=CHECK-REPEAT

; RUN: opt -passes=debugify,check-debugify -S -o - < %s | \
; RUN:   FileCheck %s -implicit-check-not="CheckModuleDebugify: FAIL"
; RUN: opt -passes=debugify,check-debugify -S -o - < %s | \
; RUN:   FileCheck %s -implicit-check-not="CheckModuleDebugify: FAIL"
; RUN: opt -enable-debugify -passes=verify -S -o - < %s | \
; RUN:   FileCheck %s -implicit-check-not="CheckModuleDebugify: FAIL"

; RUN: opt -passes=debugify,strip,check-debugify -S -o - < %s 2>&1 | \
; RUN:   FileCheck %s -check-prefix=CHECK-WARN

; RUN: opt -enable-debugify -passes=strip -S -o - < %s 2>&1 | \
; RUN:   FileCheck %s -check-prefix=CHECK-WARN

; RUN: opt -enable-debugify -S -o - < %s 2>&1 | FileCheck %s -check-prefix=PASS

; Verify that debugify can be safely used with piping
; RUN: opt -enable-debugify -O1 < %s | opt -O2 -o /dev/null
; RUN: opt -passes=debugify,mem2reg,check-debugify < %s | opt -O2 -o /dev/null

;; Perform the same checks again for intrinsic debug info
; RUN: opt --experimental-debuginfo-iterators=false -passes=debugify -S -o - < %s | FileCheck %s
; RUN: opt --experimental-debuginfo-iterators=false -passes=debugify -S -o - < %s | FileCheck %s

; RUN: opt --experimental-debuginfo-iterators=false -passes=debugify,debugify -S -o - < %s 2>&1 | \
; RUN:   FileCheck %s -check-prefix=CHECK-REPEAT
; RUN: opt --experimental-debuginfo-iterators=false -passes=debugify,debugify -S -o - < %s 2>&1 | \
; RUN:   FileCheck %s -check-prefix=CHECK-REPEAT

; RUN: opt --experimental-debuginfo-iterators=false -passes=debugify,check-debugify -S -o - < %s | \
; RUN:   FileCheck %s -implicit-check-not="CheckModuleDebugify: FAIL"
; RUN: opt --experimental-debuginfo-iterators=false -passes=debugify,check-debugify -S -o - < %s | \
; RUN:   FileCheck %s -implicit-check-not="CheckModuleDebugify: FAIL"
; RUN: opt --experimental-debuginfo-iterators=false -enable-debugify -passes=verify -S -o - < %s | \
; RUN:   FileCheck %s -implicit-check-not="CheckModuleDebugify: FAIL"

; RUN: opt --experimental-debuginfo-iterators=false -passes=debugify,strip,check-debugify -S -o - < %s 2>&1 | \
; RUN:   FileCheck %s -check-prefix=CHECK-WARN

; RUN: opt --experimental-debuginfo-iterators=false -enable-debugify -passes=strip -S -o - < %s 2>&1 | \
; RUN:   FileCheck %s -check-prefix=CHECK-WARN

; RUN: opt --experimental-debuginfo-iterators=false -enable-debugify -S -o - < %s 2>&1 | FileCheck %s -check-prefix=PASS

; RUN: opt --experimental-debuginfo-iterators=false -enable-debugify -O1 < %s | opt -O2 -o /dev/null
; RUN: opt --experimental-debuginfo-iterators=false -passes=debugify,mem2reg,check-debugify < %s | opt -O2 -o /dev/null

; CHECK-LABEL: define void @foo
define void @foo() {
; CHECK: ret void, !dbg ![[RET1:.*]]
  ret void
}

; CHECK-LABEL: define i32 @bar
define i32 @bar() {
; CHECK: call void @foo(), !dbg ![[CALL1:.*]]
  call void @foo()

; CHECK: add i32 0, 1, !dbg ![[ADD1:.*]]
  %sum = add i32 0, 1

; CHECK: ret i32 0, !dbg ![[RET2:.*]]
  ret i32 0
}

; CHECK-LABEL: define weak_odr zeroext i1 @baz
define weak_odr zeroext i1 @baz() {
; CHECK-NOT: !dbg
  ret i1 false
}

; CHECK-LABEL: define i32 @boom
define i32 @boom() {
; CHECK: [[result:%.*]] = musttail call i32 @bar(), !dbg ![[musttail:.*]]
  %retval = musttail call i32 @bar()
; CHECK-NEXT: ret i32 [[result]], !dbg ![[musttailRes:.*]]
  ret i32 %retval
}

; CHECK-DAG: !llvm.dbg.cu = !{![[CU:.*]]}
; CHECK-DAG: !llvm.debugify = !{![[NUM_INSTS:.*]], ![[NUM_VARS:.*]]}
; CHECK-DAG: "Debug Info Version"

; CHECK-DAG: ![[CU]] = distinct !DICompileUnit(language: DW_LANG_C, file: {{.*}}, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
; CHECK-DAG: !DIFile(filename: "<stdin>", directory: "/")
; CHECK-DAG: distinct !DISubprogram(name: "foo", linkageName: "foo", scope: null, file: {{.*}}, line: 1, type: {{.*}}, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: {{.*}}, retainedNodes: {{.*}})
; CHECK-DAG: distinct !DISubprogram(name: "bar", linkageName: "bar", scope: null, file: {{.*}}, line: 2, type: {{.*}}, scopeLine: 2, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: {{.*}}, retainedNodes: {{.*}})

; --- DILocations
; CHECK-DAG: ![[RET1]] = !DILocation(line: 1, column: 1
; CHECK-DAG: ![[CALL1]] = !DILocation(line: 2, column: 1
; CHECK-DAG: ![[ADD1]] = !DILocation(line: 3, column: 1
; CHECK-DAG: ![[RET2]] = !DILocation(line: 4, column: 1
; CHECK-DAG: ![[musttail]] = !DILocation(line: 5, column: 1
; CHECK-DAG: ![[musttailRes]] = !DILocation(line: 6, column: 1

; --- DILocalVariables
; CHECK-DAG: ![[TY32:.*]] = !DIBasicType(name: "ty32", size: 32, encoding: DW_ATE_unsigned)
; CHECK-DAG: !DILocalVariable(name: "1", scope: {{.*}}, file: {{.*}}, line: 1, type: ![[TY32]])
; CHECK-DAG: !DILocalVariable(name: "2", scope: {{.*}}, file: {{.*}}, line: 3, type: ![[TY32]])
; CHECK-DAG: !DILocalVariable(name: "3", scope: {{.*}}, file: {{.*}}, line: 5, type: ![[TY32]])

; --- Metadata counts
; CHECK-DAG: ![[NUM_INSTS]] = !{i32 6}
; CHECK-DAG: ![[NUM_VARS]] = !{i32 3}

; --- Repeat case
; CHECK-REPEAT: ModuleDebugify: Skipping module with debug info

; --- Failure case
; CHECK-WARN: WARNING: Instruction with empty DebugLoc in function foo --   ret void
; CHECK-WARN: WARNING: Instruction with empty DebugLoc in function bar --   call void @foo()
; CHECK-WARN: WARNING: Instruction with empty DebugLoc in function bar --   {{.*}} add i32 0, 1
; CHECK-WARN: WARNING: Instruction with empty DebugLoc in function bar --   ret i32 0
; CHECK-WARN: WARNING: Missing line 1
; CHECK-WARN: WARNING: Missing line 2
; CHECK-WARN: WARNING: Missing line 3
; CHECK-WARN: WARNING: Missing line 4
; CHECK-WARN: WARNING: Missing variable 1
; CHECK-WARN: CheckModuleDebugify: PASS

; PASS: CheckModuleDebugify: PASS
