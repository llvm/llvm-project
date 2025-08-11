; RUN: opt -S -dxil-translate-metadata < %s | FileCheck %s
; RUN: opt -S --passes="dxil-pretty-printer" < %s 2>&1 | FileCheck %s --check-prefix=PRINT
; RUN: llc %s --filetype=asm -o - 2>&1 | FileCheck %s --check-prefixes=CHECK,PRINT

target datalayout = "e-m:e-p:32:32-i1:32-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-n8:16:32:64"
target triple = "dxil-pc-shadermodel6.6-compute"

@Zero.str = private unnamed_addr constant [5 x i8] c"Zero\00", align 1
@One.str = private unnamed_addr constant [4 x i8] c"One\00", align 1
@Two.str = private unnamed_addr constant [4 x i8] c"Two\00", align 1
@Three.str = private unnamed_addr constant [6 x i8] c"Three\00", align 1
@Four.str = private unnamed_addr constant [5 x i8] c"Four\00", align 1
@Five.str = private unnamed_addr constant [5 x i8] c"Five\00", align 1
@Six.str = private unnamed_addr constant [4 x i8] c"Six\00", align 1
@Seven.str = private unnamed_addr constant [6 x i8] c"Seven\00", align 1
@Array.str = private unnamed_addr constant [6 x i8] c"Array\00", align 1
@Array2.str = private unnamed_addr constant [7 x i8] c"Array2\00", align 1

; PRINT:; Resource Bindings:
; PRINT-NEXT:;
; PRINT-NEXT:; Name                                 Type  Format         Dim      ID      HLSL Bind     Count
; PRINT-NEXT:; ------------------------------ ---------- ------- ----------- ------- -------------- ---------
; PRINT-NEXT:; Zero                              texture     f16         buf      T0             t0         1
; PRINT-NEXT:; One                               texture     f32         buf      T1             t1         1
; PRINT-NEXT:; Two                               texture     f64         buf      T2             t2         1
; PRINT-NEXT:; Three                             texture     i32         buf      T3             t3         1
; PRINT-NEXT:; Four                              texture    byte         r/o      T4             t5         1
; PRINT-NEXT:; Five                              texture  struct         r/o      T5             t6         1
; PRINT-NEXT:; Six                               texture     u64         buf      T6     t10,space2         1
; PRINT-NEXT:; Array                             texture     f32         buf      T7      t4,space3       100
; PRINT-NEXT:; Array2                            texture     f64         buf      T8      t2,space4 unbounded
; PRINT-NEXT:; Seven                             texture     u64         buf      T9     t20,space5         1
;

define void @test() #0 {
  ; Buffer<half4> Zero : register(t0)
  %Zero_h = call target("dx.TypedBuffer", <4 x half>, 0, 0, 0)
            @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, i1 false, ptr @Zero.str)
 
  ; Buffer<float4> One : register(t1)
  %One_h = call target("dx.TypedBuffer", <2 x float>, 0, 0, 0)
            @llvm.dx.resource.handlefrombinding(i32 0, i32 1, i32 1, i32 0, i1 false, ptr @One.str)
 
  ; Buffer<double> Two : register(t2);
  %Two_h = call target("dx.TypedBuffer", double, 0, 0, 0)
            @llvm.dx.resource.handlefrombinding(i32 0, i32 2, i32 1, i32 0, i1 false, ptr @Two.str)

  ; Buffer<int4> Three : register(t3);
  %Three_h = call target("dx.TypedBuffer", <4 x i32>, 0, 0, 1)
            @llvm.dx.resource.handlefrombinding(i32 0, i32 3, i32 1, i32 0, i1 false, ptr @Three.str)

  ; ByteAddressBuffer Four : register(t4)
  %Four_h = call target("dx.RawBuffer", i8, 0, 0)
            @llvm.dx.resource.handlefrombinding(i32 0, i32 5, i32 1, i32 0, i1 false, ptr @Four.str)

  ; StructuredBuffer<int16_t> Five : register(t6);
  %Five_h = call target("dx.RawBuffer", i16, 0, 0)
            @llvm.dx.resource.handlefrombinding(i32 0, i32 6, i32 1, i32 0, i1 false, ptr @Five.str)
  
  ; Buffer<double> Six : register(t10, space2);
  %Six_h = call target("dx.TypedBuffer", i64, 0, 0, 0)
            @llvm.dx.resource.handlefrombinding(i32 2, i32 10, i32 1, i32 0, i1 false, ptr @Six.str)

  ; Same buffer type as Six - should have the same type in metadata
  ; Buffer<double> Seven : register(t20, space5);
  %Seven_h = call target("dx.TypedBuffer", i64, 0, 0, 0)
            @llvm.dx.resource.handlefrombinding(i32 5, i32 20, i32 1, i32 0, i1 false, ptr @Seven.str)

  ; Buffer<float4> Array[100] : register(t4, space3);
  ; Buffer<float4> B1 = Array[30];
  ; Buffer<float4> B2 = Array[42];
  ; resource array accesses should produce one metadata entry   
  %Array_30_h = call target("dx.TypedBuffer", <4 x float>, 0, 0, 0)
            @llvm.dx.resource.handlefrombinding(i32 3, i32 4, i32 100, i32 30, i1 false, ptr @Array.str)
  %Array_42_h = call target("dx.TypedBuffer", <4 x float>, 0, 0, 0)
            @llvm.dx.resource.handlefrombinding(i32 3, i32 4, i32 100, i32 42, i1 false, ptr @Array.str)

  ; test unbounded resource array
  ; Buffer<double> Array2[] : register(t2, space4);
  ; Buffer<double> C1 = Array[10];
  ; Buffer<double> C2 = Array[20];
  %Array2_10_h = call target("dx.TypedBuffer", double, 0, 0, 0)
            @llvm.dx.resource.handlefrombinding(i32 4, i32 2, i32 -1, i32 10, i1 false, ptr @Array2.str)
  %Array2_20_h = call target("dx.TypedBuffer", double, 0, 0, 0)
            @llvm.dx.resource.handlefrombinding(i32 4, i32 2, i32 -1, i32 20, i1 false, ptr @Array2.str)

  ret void
}

attributes #0 = { noinline nounwind "hlsl.shader"="compute" }

; CHECK: %"Buffer<half4>" = type { <4 x half> }
; CHECK: %"Buffer<float2>" = type { <2 x float> }
; CHECK: %"Buffer<double>" = type { double }
; CHECK: %"Buffer<int32_t4>" = type { <4 x i32> }
; CHECK: %ByteAddressBuffer = type { i32 }
; CHECK: %"StructuredBuffer<int16_t>" = type { i16 }
; CHECK: %"Buffer<uint32_t>" = type { i64 }
; CHECK: %"Buffer<float4>" = type { <4 x float> }

; CHECK: @Zero = external constant %"Buffer<half4>"
; CHECK: @One = external constant %"Buffer<float2>"
; CHECK: @Two = external constant %"Buffer<double>"
; CHECK: @Three = external constant %"Buffer<int32_t4>"
; CHECK: @Four = external constant %ByteAddressBuffer
; CHECK: @Five = external constant %"StructuredBuffer<int16_t>"
; CHECK: @Six = external constant %"Buffer<uint32_t>"
; CHECK: @Array = external constant %"Buffer<float4>"
; CHECK: @Array2 = external constant %"Buffer<double>"
; CHECK: @Seven = external constant %"Buffer<uint32_t>"

; CHECK: !dx.resources = !{[[ResList:[!][0-9]+]]}

; CHECK: [[ResList]] = !{[[SRVList:[!][0-9]+]], null, null, null}
; CHECK: [[SRVList]] = !{![[Zero:[0-9]+]], ![[One:[0-9]+]], ![[Two:[0-9]+]],
; CHECK-SAME: ![[Three:[0-9]+]], ![[Four:[0-9]+]], ![[Five:[0-9]+]],
; CHECK-SAME: ![[Six:[0-9]+]], ![[Array:[0-9]+]], ![[Array2:[0-9]+]], ![[Seven:[0-9]+]]}

; CHECK: ![[Zero]] = !{i32 0, ptr @Zero, !"Zero", i32 0, i32 0, i32 1, i32 10, i32 0, ![[Half:[0-9]+]]}
; CHECK: ![[Half]] = !{i32 0, i32 8}
; CHECK: ![[One]] = !{i32 1, ptr @One, !"One", i32 0, i32 1, i32 1, i32 10, i32 0, ![[Float:[0-9]+]]}
; CHECK: ![[Float]] = !{i32 0, i32 9}
; CHECK: ![[Two]] = !{i32 2, ptr @Two, !"Two", i32 0, i32 2, i32 1, i32 10, i32 0, ![[Double:[0-9]+]]}
; CHECK: ![[Double]] = !{i32 0, i32 10}
; CHECK: ![[Three]] = !{i32 3, ptr @Three, !"Three", i32 0, i32 3, i32 1, i32 10, i32 0, ![[I32:[0-9]+]]}
; CHECK: ![[I32]] = !{i32 0, i32 4}
; CHECK: ![[Four]] = !{i32 4, ptr @Four, !"Four", i32 0, i32 5, i32 1, i32 11, i32 0, null}
; CHECK: ![[Five]] = !{i32 5, ptr @Five, !"Five", i32 0, i32 6, i32 1, i32 12, i32 0, ![[FiveStride:[0-9]+]]}
; CHECK: ![[FiveStride]] = !{i32 1, i32 2}
; CHECK: ![[Six]] = !{i32 6, ptr @Six, !"Six", i32 2, i32 10, i32 1, i32 10, i32 0, ![[U64:[0-9]+]]}
; CHECK: ![[U64]] = !{i32 0, i32 7}
; CHECK: ![[Array]] = !{i32 7, ptr @Array, !"Array", i32 3, i32 4, i32 100, i32 10, i32 0, ![[Float]]}
; CHECK: ![[Array2]] = !{i32 8, ptr @Array2, !"Array2", i32 4, i32 2, i32 -1, i32 10, i32 0, ![[Double]]}
; CHECK: ![[Seven]] = !{i32 9, ptr @Seven, !"Seven", i32 5, i32 20, i32 1, i32 10, i32 0, ![[U64]]}
