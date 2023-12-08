; Tests that debug information is sane after coro-split
; RUN: opt < %s -passes='cgscc(coro-split),simplifycfg,early-cse' -S | FileCheck %s

source_filename = "simple-repro.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind
define ptr @f(i32 %x) #0 personality i32 0 !dbg !6 {
entry:
  %x.addr = alloca i32, align 4
  %coro_hdl = alloca ptr, align 8
  store i32 %x, ptr %x.addr, align 4
  %0 = call token @llvm.coro.id(i32 0, ptr null, ptr @f, ptr null), !dbg !16
  %1 = call i64 @llvm.coro.size.i64(), !dbg !16
  %call = call ptr @malloc(i64 %1), !dbg !16
  %2 = call ptr @llvm.coro.begin(token %0, ptr %call) #7, !dbg !16
  store ptr %2, ptr %coro_hdl, align 8, !dbg !16
  %3 = call i8 @llvm.coro.suspend(token none, i1 false), !dbg !17
  %conv = sext i8 %3 to i32, !dbg !17
  %late_local = alloca i32, align 4
  call void @coro.devirt.trigger(ptr null)
  switch i32 %conv, label %sw.default [
    i32 0, label %sw.bb
    i32 1, label %sw.bb1
  ], !dbg !17

sw.bb:                                            ; preds = %entry
  %direct = load i32, ptr %x.addr, align 4, !dbg !14
  %gep = getelementptr inbounds [16 x i8], ptr undef, i32 %direct, !dbg !14
  call void @llvm.dbg.declare(metadata ptr %gep, metadata !27, metadata !13), !dbg !14
  call void @llvm.dbg.declare(metadata i32 %conv, metadata !26, metadata !13), !dbg !14
  call void @llvm.dbg.declare(metadata i32 %direct, metadata !25, metadata !13), !dbg !14
  call void @llvm.dbg.declare(metadata ptr %x.addr, metadata !12, metadata !13), !dbg !14
  call void @llvm.dbg.declare(metadata ptr %coro_hdl, metadata !15, metadata !13), !dbg !16
  call void @llvm.dbg.declare(metadata ptr %late_local, metadata !29, metadata !13), !dbg !16
  call void @llvm.dbg.value(metadata i32 %direct, metadata !30, metadata !13), !dbg !14
  ; don't crash when encountering nonsensical debug info, verfifier doesn't yet reject these
  call void @llvm.dbg.declare(metadata ptr null, metadata !28, metadata !13), !dbg !16
  call void @llvm.dbg.declare(metadata !{}, metadata !28, metadata !13), !dbg !16
  %new_storgae = invoke ptr @allocate()
    to label %next unwind label %ehcleanup, !dbg !18

next:
  br label %sw.epilog, !dbg !18

sw.bb1:                                           ; preds = %entry
  br label %coro_Cleanup, !dbg !18

sw.default:                                       ; preds = %entry
  br label %coro_Suspend, !dbg !18

sw.epilog:                                        ; preds = %sw.bb
  call void @llvm.dbg.declare(metadata ptr %new_storgae, metadata !31, metadata !13), !dbg !16
  %4 = load i32, ptr %x.addr, align 4, !dbg !20
  %add = add nsw i32 %4, 1, !dbg !21
  store i32 %add, ptr %x.addr, align 4, !dbg !22
  %asm_res = callbr i32 asm "", "=r,r,!i"(i32 %x)
          to label %coro_Cleanup [label %indirect.dest]

indirect.dest:
  call void @log(), !dbg !18
  br label %coro_Cleanup

coro_Cleanup:                                     ; preds = %sw.epilog, %sw.bb1
  %5 = load ptr, ptr %coro_hdl, align 8, !dbg !24
  %6 = call ptr @llvm.coro.free(token %0, ptr %5), !dbg !24
  call void @free(ptr %6), !dbg !24
  call void @llvm.dbg.declare(metadata i32 %asm_res, metadata !32, metadata !13), !dbg !16
  br label %coro_Suspend, !dbg !24

coro_Suspend:                                     ; preds = %coro_Cleanup, %sw.default
  %7 = call i1 @llvm.coro.end(ptr null, i1 false, token none) #7, !dbg !24
  %8 = load ptr, ptr %coro_hdl, align 8, !dbg !24
  store i32 0, ptr %late_local, !dbg !24
  ret ptr %8, !dbg !24

ehcleanup:
  %ex = landingpad { ptr, i32 }
          catch ptr null
  call void @print({ ptr, i32 } %ex)
  unreachable
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: argmemonly nounwind readonly
declare token @llvm.coro.id(i32, ptr readnone, ptr nocapture readonly, ptr) #2

declare ptr @malloc(i64) #3
declare ptr @allocate() 
declare void @print({ ptr, i32 })
declare void @log()

; Function Attrs: nounwind readnone
declare i64 @llvm.coro.size.i64() #4

; Function Attrs: nounwind
declare ptr @llvm.coro.begin(token, ptr writeonly) #5

; Function Attrs: nounwind
declare i8 @llvm.coro.suspend(token, i1) #5

declare void @free(ptr) #3

; Function Attrs: argmemonly nounwind readonly
declare ptr @llvm.coro.free(token, ptr nocapture readonly) #2

; Function Attrs: nounwind
declare i1 @llvm.coro.end(ptr, i1, token) #5

; Function Attrs: alwaysinline
define private void @coro.devirt.trigger(ptr) #6 {
entry:
  ret void
}

; Function Attrs: argmemonly nounwind readonly
declare ptr @llvm.coro.subfn.addr(ptr nocapture readonly, i8) #2

attributes #0 = { noinline nounwind presplitcoroutine "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { argmemonly nounwind readonly }
attributes #3 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind readnone }
attributes #5 = { nounwind }
attributes #6 = { alwaysinline }
attributes #7 = { noduplicate }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 5.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "<stdin>", directory: "C:\5CGitHub\5Cllvm\5Cbuild\5CDebug\5Cbin")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{!"clang version 5.0.0"}
!6 = distinct !DISubprogram(name: "f", linkageName: "flink", scope: !7, file: !7, line: 55, type: !8, isLocal: false, isDefinition: true, scopeLine: 55, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2, declaration: !DISubprogram(name: "f", linkageName: "flink", scope: !7, file: !7, line: 55, type: !8, isLocal: false, isDefinition: false, flags: DIFlagPrototyped))
!7 = !DIFile(filename: "simple-repro.c", directory: "C:\5CGitHub\5Cllvm\5Cbuild\5CDebug\5Cbin")
!8 = !DISubroutineType(types: !9)
!9 = !{!10, !11}
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !DILocalVariable(name: "x", arg: 1, scope: !6, file: !7, line: 55, type: !11)
!13 = !DIExpression()
!14 = !DILocation(line: 55, column: 13, scope: !6)
!15 = !DILocalVariable(name: "coro_hdl", scope: !6, file: !7, line: 56, type: !10)
!16 = !DILocation(line: 56, column: 3, scope: !6)
!17 = !DILocation(line: 58, column: 5, scope: !6)
!18 = !DILocation(line: 58, column: 5, scope: !19)
!19 = distinct !DILexicalBlock(scope: !6, file: !7, line: 58, column: 5)
!20 = !DILocation(line: 59, column: 9, scope: !6)
!21 = !DILocation(line: 59, column: 10, scope: !6)
!22 = !DILocation(line: 59, column: 7, scope: !6)
!23 = !DILocation(line: 59, column: 5, scope: !6)
!24 = !DILocation(line: 62, column: 3, scope: !6)
; These variables were added manually.
!25 = !DILocalVariable(name: "direct_mem", scope: !6, file: !7, line: 55, type: !11)
!26 = !DILocalVariable(name: "direct_const", scope: !6, file: !7, line: 55, type: !11)
!27 = !DILocalVariable(name: "undefined", scope: !6, file: !7, line: 55, type: !11)
!28 = !DILocalVariable(name: "null", scope: !6, file: !7, line: 55, type: !11)
!29 = !DILocalVariable(name: "partial_dead", scope: !6, file: !7, line: 55, type: !11)
!30 = !DILocalVariable(name: "direct_value", scope: !6, file: !7, line: 55, type: !11)
!31 = !DILocalVariable(name: "allocated", scope: !6, file: !7, line: 55, type: !11)
!32 = !DILocalVariable(name: "inline_asm", scope: !6, file: !7, line: 55, type: !11)

; CHECK: define ptr @f(i32 %x) #0 personality i32 0 !dbg ![[ORIG:[0-9]+]]
; CHECK: define internal fastcc void @f.resume(ptr noundef nonnull align 8 dereferenceable(40) %0) #0 personality i32 0 !dbg ![[RESUME:[0-9]+]]
; CHECK: entry.resume:
; CHECK: %[[DBG_PTR:.*]] = alloca ptr
; CHECK: call void @llvm.dbg.declare(metadata ptr %[[DBG_PTR]], metadata ![[RESUME_COROHDL:[0-9]+]], metadata !DIExpression(DW_OP_deref, DW_OP_plus_uconst,
; CHECK: call void @llvm.dbg.declare(metadata ptr %[[DBG_PTR]], metadata ![[RESUME_X:[0-9]+]], metadata !DIExpression(DW_OP_deref, DW_OP_plus_uconst, [[EXPR_TAIL:.*]])
; CHECK: call void @llvm.dbg.declare(metadata ptr %[[DBG_PTR]], metadata ![[RESUME_DIRECT:[0-9]+]], metadata !DIExpression(DW_OP_deref, DW_OP_plus_uconst, [[EXPR_TAIL]])
; CHECK: store ptr {{.*}}, ptr %[[DBG_PTR]]
; CHECK-NOT: alloca ptr
; CHECK: call void @llvm.dbg.declare(metadata i8 0, metadata ![[RESUME_CONST:[0-9]+]], metadata !DIExpression(DW_OP_LLVM_convert, 8, DW_ATE_signed, DW_OP_LLVM_convert, 32, DW_ATE_signed))
; Note that keeping the undef value here could be acceptable, too.
; CHECK-NOT: call void @llvm.dbg.declare(metadata ptr undef, metadata !{{[0-9]+}}, metadata !DIExpression())
; CHECK: call void @coro.devirt.trigger(ptr null)
; CHECK: call void @llvm.dbg.value(metadata ptr {{.*}}, metadata ![[RESUME_DIRECT_VALUE:[0-9]+]], metadata !DIExpression(DW_OP_deref, DW_OP_plus_uconst, {{[0-9]+}}, DW_OP_deref))
; Check that the dbg.declare intrinsic of invoke instruction is hanled correctly.
; CHECK: %[[ALLOCATED_STORAGE:.+]] = invoke ptr @allocate()
; CHECK-NEXT: to label %[[NORMAL_DEST:.+]] unwind
; CHECK: [[NORMAL_DEST]]
; CHECK-NEXT: call void @llvm.dbg.declare(metadata ptr %[[ALLOCATED_STORAGE]]
; CHECK: %[[CALLBR_RES:.+]] = callbr i32 asm
; CHECK-NEXT: to label %[[DEFAULT_DEST:.+]] [label
; CHECK: [[DEFAULT_DEST]]:
; CHECK-NOT: {{.*}}:
; CHECK: call void @llvm.dbg.declare(metadata i32 %[[CALLBR_RES]]
; CHECK: define internal fastcc void @f.destroy(ptr noundef nonnull align 8 dereferenceable(40) %0) #0 personality i32 0 !dbg ![[DESTROY:[0-9]+]]
; CHECK: define internal fastcc void @f.cleanup(ptr noundef nonnull align 8 dereferenceable(40) %0) #0 personality i32 0 !dbg ![[CLEANUP:[0-9]+]]

; CHECK: ![[ORIG]] = distinct !DISubprogram(name: "f", linkageName: "flink"

; CHECK: ![[RESUME]] = distinct !DISubprogram(name: "f", linkageName: "flink"
; CHECK: ![[RESUME_COROHDL]] = !DILocalVariable(name: "coro_hdl", scope: ![[RESUME]]
; CHECK: ![[RESUME_X]] = !DILocalVariable(name: "x", arg: 1, scope: ![[RESUME]]
; CHECK: ![[RESUME_DIRECT]] = !DILocalVariable(name: "direct_mem", scope: ![[RESUME]]
; CHECK: ![[RESUME_CONST]] = !DILocalVariable(name: "direct_const", scope: ![[RESUME]]
; CHECK: ![[RESUME_DIRECT_VALUE]] = !DILocalVariable(name: "direct_value", scope: ![[RESUME]]

; CHECK: ![[DESTROY]] = distinct !DISubprogram(name: "f", linkageName: "flink"

; CHECK: ![[CLEANUP]] = distinct !DISubprogram(name: "f", linkageName: "flink"
