; RUN: opt -safe-stack -S -mtriple=i386-pc-linux-gnu < %s -o - | FileCheck %s

; Test debug location for the local variables moved onto the unsafe stack.
; CHECK: define void @f
; CHECK: %[[USP:.*]] = load i8*, i8** @__safestack_unsafe_stack_ptr

; dbg.declare for %buf is gone; replaced with dbg.declare based off the unsafe stack pointer
; CHECK-NOT: @llvm.dbg.declare.*%buf
; CHECK: call void @llvm.dbg.declare(metadata i8* %[[USP]], metadata ![[VAR:.*]], metadata ![[EXPR:.*]])

; dbg.declare appears before the first use of %buf
; CHECK: getelementptr{{.*}}%buf
; CHECK: call{{.*}}@Capture
; CHECK: ret void

; dbg.declare describes "buf"...
; CHECK: ![[VAR]] = !DILocalVariable(name: "buf"

; ... as an offset from the unsafe stack pointer
; CHECK: ![[EXPR]] = !DIExpression(DW_OP_deref, DW_OP_minus, 400)


target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: safestack uwtable
define void @f() #0 !dbg !4 {
entry:
  %buf = alloca [100 x i32], align 16
  %0 = bitcast [100 x i32]* %buf to i8*, !dbg !16
  call void @llvm.lifetime.start(i64 400, i8* %0) #4, !dbg !16
  tail call void @llvm.dbg.declare(metadata [100 x i32]* %buf, metadata !8, metadata !17), !dbg !18


  %arraydecay = getelementptr inbounds [100 x i32], [100 x i32]* %buf, i64 0, i64 0, !dbg !19
  call void @Capture(i32* %arraydecay), !dbg !20
  call void @llvm.lifetime.end(i64 400, i8* %0) #4, !dbg !21
  ret void, !dbg !21
}

; Function Attrs: nounwind argmemonly
declare void @llvm.lifetime.start(i64, i8* nocapture) #1

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #2

declare void @Capture(i32*) #3

; Function Attrs: nounwind argmemonly
declare void @llvm.lifetime.end(i64, i8* nocapture) #1

attributes #0 = { safestack uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind argmemonly }
attributes #2 = { nounwind readnone }
attributes #3 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!13, !14}
!llvm.ident = !{!15}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 3.8.0 (trunk 248518) (llvm/trunk 248512)", isOptimized: true, runtimeVersion: 0, emissionKind: 1, enums: !2, subprograms: !3)
!1 = !DIFile(filename: "1.cc", directory: "/tmp")
!2 = !{}
!3 = !{!4}
!4 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 4, type: !5, isLocal: false, isDefinition: true, scopeLine: 4, flags: DIFlagPrototyped, isOptimized: true, variables: !7)
!5 = !DISubroutineType(types: !6)
!6 = !{null}
!7 = !{!8}
!8 = !DILocalVariable(name: "buf", scope: !4, file: !1, line: 5, type: !9)
!9 = !DICompositeType(tag: DW_TAG_array_type, baseType: !10, size: 3200, align: 32, elements: !11)
!10 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!11 = !{!12}
!12 = !DISubrange(count: 100)
!13 = !{i32 2, !"Dwarf Version", i32 4}
!14 = !{i32 2, !"Debug Info Version", i32 3}
!15 = !{!"clang version 3.8.0 (trunk 248518) (llvm/trunk 248512)"}
!16 = !DILocation(line: 5, column: 3, scope: !4)
!17 = !DIExpression()
!18 = !DILocation(line: 5, column: 7, scope: !4)
!19 = !DILocation(line: 6, column: 11, scope: !4)
!20 = !DILocation(line: 6, column: 3, scope: !4)
!21 = !DILocation(line: 7, column: 1, scope: !4)
