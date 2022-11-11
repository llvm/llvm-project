; RUN: opt -annotation-remarks -o /dev/null -S -pass-remarks-output=%t.opt.yaml %s -pass-remarks-missed=annotation-remarks 2>&1 | FileCheck %s
; RUN: cat %t.opt.yaml | FileCheck -check-prefix=YAML %s

; Emit a remark that reports a store.
define void @store(ptr %dst) {
; CHECK:      Store inserted by -ftrivial-auto-var-init.
; CHECK-NEXT: Store size: 4 bytes.
; YAML-LABEL: --- !Missed
; YAML-NEXT: Pass:            annotation-remarks
; YAML-NEXT: Name:            AutoInitStore
; YAML-NEXT: DebugLoc:
; YAML-NEXT: Function:        store
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          Store inserted by -ftrivial-auto-var-init.
; YAML-NEXT:   - String:          "\nStore size: "
; YAML-NEXT:   - StoreSize:       '4'
; YAML-NEXT:   - String:          ' bytes.'
; YAML-NEXT:   - String:          ' Volatile: '
; YAML-NEXT:   - StoreVolatile:       'false'
; YAML-NEXT:   - String:          .
; YAML-NEXT:   - String:          ' Atomic: '
; YAML-NEXT:   - StoreAtomic:       'false'
; YAML-NEXT:   - String:          .
; YAML-NEXT: ...
  store i32 0, ptr %dst, !annotation !0, !dbg !DILocation(scope: !4)
  ret void
}

; Emit a remark that reports a volatile store.
define void @volatile_store(ptr %dst) {
; CHECK-NEXT: Store inserted by -ftrivial-auto-var-init.
; CHECK-NEXT: Store size: 4 bytes. Volatile: true.
; YAML-LABEL: --- !Missed
; YAML-NEXT: Pass:            annotation-remarks
; YAML-NEXT: Name:            AutoInitStore
; YAML-NEXT: DebugLoc:
; YAML-NEXT: Function:        volatile_store
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          Store inserted by -ftrivial-auto-var-init.
; YAML-NEXT:   - String:          "\nStore size: "
; YAML-NEXT:   - StoreSize:       '4'
; YAML-NEXT:   - String:          ' bytes.'
; YAML-NEXT:   - String:          ' Volatile: '
; YAML-NEXT:   - StoreVolatile:       'true'
; YAML-NEXT:   - String:          .
; YAML-NEXT:   - String:          ' Atomic: '
; YAML-NEXT:   - StoreAtomic:       'false'
; YAML-NEXT:   - String:          .
; YAML-NEXT: ...
  store volatile i32 0, ptr %dst, !annotation !0, !dbg !DILocation(scope: !4)
  ret void
}

; Emit a remark that reports an atomic store.
define void @atomic_store(ptr %dst) {
; CHECK-NEXT: Store inserted by -ftrivial-auto-var-init.
; CHECK-NEXT: Store size: 4 bytes. Atomic: true.
; YAML-LABEL: --- !Missed
; YAML-NEXT: Pass:            annotation-remarks
; YAML-NEXT: Name:            AutoInitStore
; YAML-NEXT: DebugLoc:
; YAML-NEXT: Function:        atomic_store
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          Store inserted by -ftrivial-auto-var-init.
; YAML-NEXT:   - String:          "\nStore size: "
; YAML-NEXT:   - StoreSize:       '4'
; YAML-NEXT:   - String:          ' bytes.'
; YAML-NEXT:   - String:          ' Atomic: '
; YAML-NEXT:   - StoreAtomic:       'true'
; YAML-NEXT:   - String:          .
; YAML-NEXT:   - String:          ' Volatile: '
; YAML-NEXT:   - StoreVolatile:       'false'
; YAML-NEXT:   - String:          .
; YAML-NEXT: ...
  store atomic i32 0, ptr %dst unordered, align 4, !annotation !0, !dbg !DILocation(scope: !4)
  ret void
}

; Emit a remark that reports a store to an alloca.
define void @store_alloca() {
; CHECK-NEXT: Store inserted by -ftrivial-auto-var-init.
; CHECK-NEXT: Store size: 4 bytes.
; CHECK-NEXT: Variables: dst (4 bytes).
; YAML-LABEL: --- !Missed
; YAML-NEXT: Pass:            annotation-remarks
; YAML-NEXT: Name:            AutoInitStore
; YAML-NEXT: DebugLoc:
; YAML-NEXT: Function:        store_alloca
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          Store inserted by -ftrivial-auto-var-init.
; YAML-NEXT:   - String:          "\nStore size: "
; YAML-NEXT:   - StoreSize:       '4'
; YAML-NEXT:   - String:          ' bytes.'
; YAML-NEXT:   - String:          "\n Written Variables: "
; YAML-NEXT:   - WVarName:        dst
; YAML-NEXT:   - String:          ' ('
; YAML-NEXT:   - WVarSize:        '4'
; YAML-NEXT:   - String:          ' bytes)'
; YAML-NEXT:   - String:          .
; YAML-NEXT:   - String:          ' Volatile: '
; YAML-NEXT:   - StoreVolatile:    'false'
; YAML-NEXT:   - String:          .
; YAML-NEXT:   - String:          ' Atomic: '
; YAML-NEXT:   - StoreAtomic:     'false'
; YAML-NEXT:   - String:          .
; YAML-NEXT: ...
  %dst = alloca i32
  store i32 0, ptr %dst, !annotation !0, !dbg !DILocation(scope: !4)
  ret void
}

; Emit a remark that reports a store to an alloca through a GEP.
define void @store_alloca_gep() {
; CHECK-NEXT: Store inserted by -ftrivial-auto-var-init.
; CHECK-NEXT: Store size: 4 bytes.
; CHECK-NEXT: Variables: dst (4 bytes).
  %dst = alloca i32
  store i32 0, ptr %dst, !annotation !0, !dbg !DILocation(scope: !4)
  ret void
}

; Emit a remark that reports a store to an alloca through a GEP, with ptrtoint+inttoptr in the way.
define void @store_alloca_gep_inttoptr() {
; CHECK-NEXT: Store inserted by -ftrivial-auto-var-init.
; CHECK-NEXT: Store size: 4 bytes.
; CHECK-NEXT: Variables: dst (4 bytes).
  %dst = alloca i32
  %p2i = ptrtoint ptr %dst to i64
  %i2p = inttoptr i64 %p2i to ptr
  store i32 0, ptr %i2p, !annotation !0, !dbg !DILocation(scope: !4)
  ret void
}

; Emit a remark that reports a store to an alloca through a GEP in an array.
define void @store_alloca_gep_array() {
; CHECK-NEXT: Store inserted by -ftrivial-auto-var-init.
; CHECK-NEXT: Store size: 4 bytes.
; CHECK-NEXT: Variables: dst (8 bytes).
  %dst = alloca [2 x i32]
  store i32 0, ptr %dst, !annotation !0, !dbg !DILocation(scope: !4)
  ret void
}

; Emit a remark that reports a store to an alloca through a bitcast.
define void @store_alloca_bitcast() {
; CHECK-NEXT: Store inserted by -ftrivial-auto-var-init.
; CHECK-NEXT: Store size: 4 bytes.
; CHECK-NEXT: Variables: dst (4 bytes).
  %dst = alloca [2 x i16]
  store i32 0, ptr %dst, !annotation !0, !dbg !DILocation(scope: !4)
  ret void
}

; Emit a remark that reports a store to an alloca that has a DILocalVariable
; attached.
define void @store_alloca_di() {
; CHECK-NEXT: Store inserted by -ftrivial-auto-var-init.
; CHECK-NEXT: Store size: 4 bytes.
; CHECK-NEXT: Variables: destination (4 bytes).
  %dst = alloca i32
  store i32 0, ptr %dst, !annotation !0, !dbg !DILocation(scope: !4)
  call void @llvm.dbg.declare(metadata ptr %dst, metadata !6, metadata !DIExpression()), !dbg !DILocation(scope: !4)
  ret void
}

; Emit a remark that reports a store to an alloca that has more than one
; DILocalVariable attached.
define void @store_alloca_di_multiple() {
; CHECK-NEXT: Store inserted by -ftrivial-auto-var-init.
; CHECK-NEXT: Store size: 4 bytes.
; CHECK-NEXT: Variables: destination2 (4 bytes), destination (4 bytes).
  %dst = alloca i32
  store i32 0, ptr %dst, !annotation !0, !dbg !DILocation(scope: !4)
  call void @llvm.dbg.declare(metadata ptr %dst, metadata !6, metadata !DIExpression()), !dbg !DILocation(scope: !4)
  call void @llvm.dbg.declare(metadata ptr %dst, metadata !7, metadata !DIExpression()), !dbg !DILocation(scope: !4)
  ret void
}

; Emit a remark that reports a store to a PHI node that can be two different
; allocas.
define void @store_alloca_phi() {
; CHECK-NEXT: Store inserted by -ftrivial-auto-var-init.
; CHECK-NEXT: Store size: 4 bytes.
; CHECK-NEXT: Variables: dst2 (4 bytes), dst (4 bytes).
entry:
  %dst = alloca i32
  %dst2 = alloca i32
  %cmp = icmp eq i32 undef, undef
  br i1 %cmp, label %l0, label %l1
l0:
  br label %l2
l1:
  br label %l2
l2:
  %phidst = phi ptr [ %dst, %l0 ], [ %dst2, %l1 ]
  store i32 0, ptr %phidst, !annotation !0, !dbg !DILocation(scope: !4)
  ret void
}

; Emit a remark that reports a store to a PHI node that can be two different
; allocas, where one of it has multiple DILocalVariable.
define void @store_alloca_phi_di_multiple() {
; CHECK-NEXT: Store inserted by -ftrivial-auto-var-init.
; CHECK-NEXT: Store size: 4 bytes.
; CHECK-NEXT: Variables: dst2 (4 bytes), destination2 (4 bytes), destination (4 bytes).
entry:
  %dst = alloca i32
  %dst2 = alloca i32
  call void @llvm.dbg.declare(metadata ptr %dst, metadata !6, metadata !DIExpression()), !dbg !DILocation(scope: !4)
  call void @llvm.dbg.declare(metadata ptr %dst, metadata !7, metadata !DIExpression()), !dbg !DILocation(scope: !4)
  %cmp = icmp eq i32 undef, undef
  br i1 %cmp, label %l0, label %l1
l0:
  br label %l2
l1:
  br label %l2
l2:
  %phidst = phi ptr [ %dst, %l0 ], [ %dst2, %l1 ]
  store i32 0, ptr %phidst, !annotation !0, !dbg !DILocation(scope: !4)
  ret void
}

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone speculatable willreturn

!llvm.module.flags = !{!1}
!0 = !{ !"auto-init" }
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3)
!3 = !DIFile(filename: "file", directory: "")
!4 = distinct !DISubprogram(name: "function", scope: !3, file: !3, unit: !2)
!5 = !DIBasicType(name: "int", size: 32)
!6 = !DILocalVariable(name: "destination", scope: !4, file: !3, type: !5)
!7 = !DILocalVariable(name: "destination2", scope: !4, file: !3, type: !5)
