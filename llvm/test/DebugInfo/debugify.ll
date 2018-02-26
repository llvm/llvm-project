; RUN: opt -debugify -S -o - < %s | FileCheck %s
; RUN: opt -passes=debugify -S -o - < %s | FileCheck %s

; RUN: opt -debugify -debugify -S -o - < %s 2>&1 | \
; RUN:   FileCheck %s -check-prefix=CHECK-REPEAT
; RUN: opt -passes=debugify,debugify -S -o - < %s 2>&1 | \
; RUN:   FileCheck %s -check-prefix=CHECK-REPEAT

; RUN: opt -debugify -check-debugify -S -o - < %s | \
; RUN:   FileCheck %s -implicit-check-not="CheckDebugify: FAIL"
; RUN: opt -passes=debugify,check-debugify -S -o - < %s | \
; RUN:   FileCheck %s -implicit-check-not="CheckDebugify: FAIL"
; RUN: opt -enable-debugify -passes=verify -S -o - < %s | \
; RUN:   FileCheck %s -implicit-check-not="CheckDebugify: FAIL"

; RUN: opt -debugify -strip -check-debugify -S -o - < %s | \
; RUN:   FileCheck %s -check-prefix=CHECK-FAIL

; RUN: opt -enable-debugify -strip -S -o - < %s | \
; RUN:   FileCheck %s -check-prefix=CHECK-FAIL

; RUN: opt -enable-debugify -S -o - < %s | FileCheck %s -check-prefix=PASS

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

; CHECK-DAG: !llvm.dbg.cu = !{![[CU:.*]]}
; CHECK-DAG: !llvm.debugify = !{![[NUM_INSTS:.*]], ![[NUM_VARS:.*]]}

; CHECK-DAG: ![[CU]] = distinct !DICompileUnit(language: DW_LANG_C, file: {{.*}}, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: {{.*}})
; CHECK-DAG: !DIFile(filename: "<stdin>", directory: "/")
; CHECK-DAG: distinct !DISubprogram(name: "foo", linkageName: "foo", scope: null, file: {{.*}}, line: 1, type: {{.*}}, isLocal: false, isDefinition: true, scopeLine: 1, isOptimized: true, unit: {{.*}}, variables: {{.*}})
; CHECK-DAG: distinct !DISubprogram(name: "bar", linkageName: "bar", scope: null, file: {{.*}}, line: 2, type: {{.*}}, isLocal: false, isDefinition: true, scopeLine: 2, isOptimized: true, unit: {{.*}}, variables: {{.*}})

; --- DILocations
; CHECK-DAG: ![[RET1]] = !DILocation(line: 1, column: 1
; CHECK-DAG: ![[CALL1]] = !DILocation(line: 2, column: 1
; CHECK-DAG: ![[ADD1]] = !DILocation(line: 3, column: 1
; CHECK-DAG: ![[RET2]] = !DILocation(line: 4, column: 1

; --- DILocalVariables
; CHECK-DAG: ![[TY32:.*]] = !DIBasicType(name: "ty32", size: 32, encoding: DW_ATE_unsigned)
; CHECK-DAG: !DILocalVariable(name: "1", scope: {{.*}}, file: {{.*}}, line: 3, type: ![[TY32]])

; --- Metadata counts
; CHECK-DAG: ![[NUM_INSTS]] = !{i32 4}
; CHECK-DAG: ![[NUM_VARS]] = !{i32 1}

; --- Repeat case
; CHECK-REPEAT: Debugify: Skipping module with debug info

; --- Failure case
; CHECK-FAIL: ERROR: Instruction with empty DebugLoc --   ret void
; CHECK-FAIL: ERROR: Instruction with empty DebugLoc --   call void @foo()
; CHECK-FAIL: ERROR: Instruction with empty DebugLoc --   {{.*}} add i32 0, 1
; CHECK-FAIL: ERROR: Instruction with empty DebugLoc --   ret i32 0
; CHECK-FAIL: WARNING: Missing line 1
; CHECK-FAIL: WARNING: Missing line 2
; CHECK-FAIL: WARNING: Missing line 3
; CHECK-FAIL: WARNING: Missing line 4
; CHECK-FAIL: ERROR: Missing variable 1
; CHECK-FAIL: CheckDebugify: FAIL

; PASS: CheckDebugify: PASS
