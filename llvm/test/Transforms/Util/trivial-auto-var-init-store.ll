; RUN: opt -annotation-remarks -o /dev/null -S -pass-remarks-output=%t.opt.yaml %s -pass-remarks-missed=annotation-remarks 2>&1 | FileCheck %s
; RUN: cat %t.opt.yaml | FileCheck -check-prefix=YAML %s

; Emit a remark that reports a store.
define void @store(i32* %dst) {
; CHECK: Initialization inserted by -ftrivial-auto-var-init.
; YAML-LABEL: --- !Missed
; YAML-NEXT: Pass:            annotation-remarks
; YAML-NEXT: Name:            AutoInitUnknownInstruction
; YAML-NEXT: DebugLoc:
; YAML-NEXT: Function:        store
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          Initialization inserted by -ftrivial-auto-var-init.
; YAML-NEXT: ...
  store i32 0, i32* %dst, !annotation !0, !dbg !DILocation(scope: !4)
  ret void
}

; Emit a remark that reports a volatile store.
define void @volatile_store(i32* %dst) {
; CHECK-NEXT: Initialization inserted by -ftrivial-auto-var-init.
; YAML-LABEL: --- !Missed
; YAML-NEXT: Pass:            annotation-remarks
; YAML-NEXT: Name:            AutoInitUnknownInstruction
; YAML-NEXT: DebugLoc:
; YAML-NEXT: Function:        volatile_store
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          Initialization inserted by -ftrivial-auto-var-init.
; YAML-NEXT: ...
  store volatile i32 0, i32* %dst, !annotation !0, !dbg !DILocation(scope: !4)
  ret void
}

; Emit a remark that reports an atomic store.
define void @atomic_store(i32* %dst) {
; CHECK-NEXT: Initialization inserted by -ftrivial-auto-var-init.
; YAML-LABEL: --- !Missed
; YAML-NEXT: Pass:            annotation-remarks
; YAML-NEXT: Name:            AutoInitUnknownInstruction
; YAML-NEXT: DebugLoc:
; YAML-NEXT: Function:        atomic_store
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          Initialization inserted by -ftrivial-auto-var-init.
; YAML-NEXT: ...
  store atomic i32 0, i32* %dst unordered, align 4, !annotation !0, !dbg !DILocation(scope: !4)
  ret void
}

; Emit a remark that reports a store to an alloca.
define void @store_alloca() {
; CHECK-NEXT: Initialization inserted by -ftrivial-auto-var-init.
; YAML-LABEL: --- !Missed
; YAML-NEXT: Pass:            annotation-remarks
; YAML-NEXT: Name:            AutoInitUnknownInstruction
; YAML-NEXT: DebugLoc:
; YAML-NEXT: Function:        store_alloca
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          Initialization inserted by -ftrivial-auto-var-init.
; YAML-NEXT: ...
  %dst = alloca i32
  store i32 0, i32* %dst, !annotation !0, !dbg !DILocation(scope: !4)
  ret void
}

; Emit a remark that reports a store to an alloca through a GEP.
define void @store_alloca_gep() {
; CHECK-NEXT: Initialization inserted by -ftrivial-auto-var-init.
  %dst = alloca i32
  %gep = getelementptr i32, i32* %dst, i32 0
  store i32 0, i32* %gep, !annotation !0, !dbg !DILocation(scope: !4)
  ret void
}

; Emit a remark that reports a store to an alloca through a GEP in an array.
define void @store_alloca_gep_array() {
; CHECK-NEXT: Initialization inserted by -ftrivial-auto-var-init.
  %dst = alloca [2 x i32]
  %gep = getelementptr [2 x i32], [2 x i32]* %dst, i64 0, i64 0
  store i32 0, i32* %gep, !annotation !0, !dbg !DILocation(scope: !4)
  ret void
}

; Emit a remark that reports a store to an alloca through a bitcast.
define void @store_alloca_bitcast() {
; CHECK-NEXT: Initialization inserted by -ftrivial-auto-var-init.
  %dst = alloca [2 x i16]
  %bc = bitcast [2 x i16]* %dst to i32*
  store i32 0, i32* %bc, !annotation !0, !dbg !DILocation(scope: !4)
  ret void
}

; Emit a remark that reports a store to an alloca that has a DILocalVariable
; attached.
define void @store_alloca_di() {
; CHECK-NEXT: Initialization inserted by -ftrivial-auto-var-init.
  %dst = alloca i32
  store i32 0, i32* %dst, !annotation !0, !dbg !DILocation(scope: !4)
  call void @llvm.dbg.declare(metadata i32* %dst, metadata !6, metadata !DIExpression()), !dbg !DILocation(scope: !4)
  ret void
}

; Emit a remark that reports a store to an alloca that has more than one
; DILocalVariable attached.
define void @store_alloca_di_multiple() {
; CHECK-NEXT: Initialization inserted by -ftrivial-auto-var-init.
  %dst = alloca i32
  store i32 0, i32* %dst, !annotation !0, !dbg !DILocation(scope: !4)
  call void @llvm.dbg.declare(metadata i32* %dst, metadata !6, metadata !DIExpression()), !dbg !DILocation(scope: !4)
  call void @llvm.dbg.declare(metadata i32* %dst, metadata !7, metadata !DIExpression()), !dbg !DILocation(scope: !4)
  ret void
}

; Emit a remark that reports a store to a PHI node that can be two different
; allocas.
define void @store_alloca_phi() {
; CHECK-NEXT: Initialization inserted by -ftrivial-auto-var-init.
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
  %phidst = phi i32* [ %dst, %l0 ], [ %dst2, %l1 ]
  store i32 0, i32* %phidst, !annotation !0, !dbg !DILocation(scope: !4)
  ret void
}

; Emit a remark that reports a store to a PHI node that can be two different
; allocas, where one of it has multiple DILocalVariable.
define void @store_alloca_phi_di_multiple() {
; CHECK-NEXT: Initialization inserted by -ftrivial-auto-var-init.
entry:
  %dst = alloca i32
  %dst2 = alloca i32
  call void @llvm.dbg.declare(metadata i32* %dst, metadata !6, metadata !DIExpression()), !dbg !DILocation(scope: !4)
  call void @llvm.dbg.declare(metadata i32* %dst, metadata !7, metadata !DIExpression()), !dbg !DILocation(scope: !4)
  %cmp = icmp eq i32 undef, undef
  br i1 %cmp, label %l0, label %l1
l0:
  br label %l2
l1:
  br label %l2
l2:
  %phidst = phi i32* [ %dst, %l0 ], [ %dst2, %l1 ]
  store i32 0, i32* %phidst, !annotation !0, !dbg !DILocation(scope: !4)
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
