; RUN: opt -S -dxil-translate-metadata < %s | FileCheck %s

; Ensures that only metadata explictly specified on the allow list, or debug
; related, metadata is emitted

target triple = "dxil-unknown-shadermodel6.0-compute"

; CHECK-NOT: !dx.rootsignatures
; CHECK-NOT: !llvm.errno.tbaa

; CHECK-DAG: !llvm.dbg.cu

; CHECK-DAG: !llvm.module.flags = !{![[#DWARF_VER:]], ![[#DEBUG_VER:]]}
; CHECK-DAG: !llvm.ident = !{![[#IDENT:]]}

; CHECK-DAG: !dx.shaderModel
; CHECK-DAG: !dx.version
; CHECK-DAG: !dx.entryPoints
; CHECK-DAG: !dx.valver
; CHECK-DAG: !dx.resources

; CHECK-NOT: !dx.rootsignatures
; CHECK-NOT: !llvm.errno.tbaa

; Check allowed llvm metadata structure to ensure it is still DXIL compatible
; If this fails, please ensure that the updated form is DXIL compatible before
; updating the test.

; CHECK-DAG: ![[#IDENT]] = !{!"clang 22.0.0"}
; CHECK-DAG: ![[#DWARF_VER]] = !{i32 2, !"Dwarf Version", i32 2}
; CHECK-DAG: ![[#DEBUG_VER]] = !{i32 2, !"Debug Info Version", i32 3}

; CHECK-NOT: !dx.rootsignatures
; CHECK-NOT: !llvm.errno.tbaa

@BufA.str = private unnamed_addr constant [5 x i8] c"BufA\00", align 1

define void @main () #0 {
entry:
  %typed0 = call target("dx.TypedBuffer", <4 x float>, 1, 0, 0)
              @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_v4f32_1_0_0(
                  i32 3, i32 5, i32 1, i32 0, ptr @BufA.str)
  ret void
}

attributes #0 = { noinline nounwind "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }

; Incompatible
!dx.rootsignatures = !{!2}
!llvm.errno.tbaa = !{!5}

; Compatible
!llvm.dbg.cu = !{!8}
!llvm.module.flags = !{!11, !12}
!llvm.ident = !{!13}
!dx.valver = !{!14}

!2 = !{ ptr @main, !3, i32 2 }
!3 = !{ !4 }
!4 = !{ !"RootFlags", i32 1 }

!5 = !{!6, !6, i64 0}
!6 = !{!"omnipotent char", !7}
!7 = !{!"Simple C/C++ TBAA"}

!8 = distinct !DICompileUnit(language: DW_LANG_C99, file: !9, producer: "Some Compiler", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !10, splitDebugInlining: false, nameTableKind: None)
!9 = !DIFile(filename: "hlsl.hlsl", directory: "/some-path")
!10 = !{}

!11 = !{i32 2, !"Dwarf Version", i32 2}
!12 = !{i32 2, !"Debug Info Version", i32 3}

!13 = !{!"clang 22.0.0"}

!14 = !{i32 1, i32 1}
