; RUN: opt -S -passes=globaldce < %s | FileCheck %s

; Test that the presence of debug intrinsics isn't affecting GlobalDCE.
; CHECK: @llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @_notremovable, i8* null }]
; CHECK-NOT: @_GLOBAL__I_a

declare void @_notremovable()

@llvm.global_ctors = appending global [3 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__I_a, i8* null }, { i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__I_b, i8* null }, { i32, void ()*, i8* } { i32 65535, void ()* @_notremovable, i8* null }]

@x = internal unnamed_addr constant i8 undef, align 1

; Function Attrs: nounwind readnone
define internal void @_GLOBAL__I_a() #1 section "__TEXT,__StaticInit,regular,pure_instructions" {
entry:
  ret void
}

; Function Attrs: nounwind readnone
define internal void @_GLOBAL__I_b() #1 section "__TEXT,__StaticInit,regular,pure_instructions" {
entry:
  tail call void @llvm.dbg.value(metadata i8* @x, metadata !4, metadata !DIExpression(DW_OP_deref, DW_OP_constu, 0, DW_OP_swap, DW_OP_xderef)), !dbg !5
  ret void
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.module.flags = !{!0}
!llvm.dbg.cu = !{!1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !2, emissionKind: FullDebug)
!2 = !DIFile(filename: "filename", directory: "directory")
!3 = distinct !DISubprogram(name: "h1", unit: !1)
!4 = !DILocalVariable(name: "b", arg: 1, scope: !3)
!5 = !DILocation(scope: !3)
