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
@Eight.str = private unnamed_addr constant [6 x i8] c"Eight\00", align 1
@Nine.str = private unnamed_addr constant [5 x i8] c"Nine\00", align 1
@Ten.str = private unnamed_addr constant [4 x i8] c"Ten\00", align 1
@Array.str = private unnamed_addr constant [6 x i8] c"Array\00", align 1
@Array2.str = private unnamed_addr constant [7 x i8] c"Array2\00", align 1

; PRINT:; Resource Bindings:
; PRINT-NEXT:;
; PRINT-NEXT:; Name                                 Type  Format         Dim      ID      HLSL Bind     Count
; PRINT-NEXT:; ------------------------------ ---------- ------- ----------- ------- -------------- ---------
; PRINT-NEXT:; Zero                                  UAV     f16         buf      U0             u0         1
; PRINT-NEXT:; One                                   UAV     f32         buf      U1             u1         1
; PRINT-NEXT:; Two                                   UAV     f64         buf      U2             u2         1
; PRINT-NEXT:; Three                                 UAV     i32         buf      U3             u3         1
; PRINT-NEXT:; Four                                  UAV    byte         r/w      U4             u5         1
; PRINT-NEXT:; Five                                  UAV  struct         r/w      U5             u6         1
; PRINT-NEXT:; Six                                   UAV     i32         buf      U6             u7         1
; PRINT-NEXT:; Seven                                 UAV  struct         r/w      U7             u8         1
; PRINT-NEXT:; Eight                                 UAV    byte         r/w      U8             u9         1
; PRINT-NEXT:; Nine                                  UAV     u64         buf      U9     u10,space2         1
; PRINT-NEXT:; Array                                 UAV     f32         buf     U10      u4,space3       100
; PRINT-NEXT:; Array2                                UAV     f64         buf     U11      u2,space4 unbounded
; PRINT-NEXT:; Ten                                   UAV     u64         buf     U12     u22,space5         1

define void @test() #0 {
  ; RWBuffer<half4> Zero : register(u0)
  %Zero_h = call target("dx.TypedBuffer", <4 x half>, 1, 0, 0)
            @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, i1 false, ptr @Zero.str)
 
  ; RWBuffer<float4> One : register(u1)
  %One_h = call target("dx.TypedBuffer", <2 x float>, 1, 0, 0)
            @llvm.dx.resource.handlefrombinding(i32 0, i32 1, i32 1, i32 0, i1 false, ptr @One.str)
 
  ; RWBuffer<double> Two : register(u2);
  %Two_h = call target("dx.TypedBuffer", double, 1, 0, 0)
            @llvm.dx.resource.handlefrombinding(i32 0, i32 2, i32 1, i32 0, i1 false, ptr @Two.str)

  ; RWBuffer<int4> Three : register(u3);
  %Three_h = call target("dx.TypedBuffer", <4 x i32>, 1, 0, 1)
            @llvm.dx.resource.handlefrombinding(i32 0, i32 3, i32 1, i32 0, i1 false, ptr @Three.str)

  ; ByteAddressBuffer Four : register(u5)
  %Four_h = call target("dx.RawBuffer", i8, 1, 0)
            @llvm.dx.resource.handlefrombinding(i32 0, i32 5, i32 1, i32 0, i1 false, ptr @Four.str)

  ; RWStructuredBuffer<int16_t> Five : register(u6);
  %Five_h = call target("dx.RawBuffer", i16, 1, 0)
            @llvm.dx.resource.handlefrombinding(i32 0, i32 6, i32 1, i32 0, i1 false, ptr @Five.str)
  
  ; RasterizerOrderedBuffer<int4> Six : register(u7);
  %Six_h = call target("dx.TypedBuffer", <4 x i32>, 1, 1, 1)
            @llvm.dx.resource.handlefrombinding(i32 0, i32 7, i32 1, i32 0, i1 false, ptr @Six.str)

  ; RasterizerOrderedStructuredBuffer<uint4> Seven : register(u3, space10);
  %Seven_h = call target("dx.RawBuffer", <4 x i32>, 1, 1)
            @llvm.dx.resource.handlefrombinding(i32 0, i32 8, i32 1, i32 0, i1 false, ptr @Seven.str)

  ; RasterizerOrderedByteAddressBuffer Eight : register(u9); 
  %Eight_h = call target("dx.RawBuffer", i8, 1, 1) 
            @llvm.dx.resource.handlefrombinding(i32 0, i32 9, i32 1, i32 0, i1 false, ptr @Eight.str)

  ; RWBuffer<double> Nine : register(u2);
  %Nine_h = call target("dx.TypedBuffer", i64, 1, 0, 0)
            @llvm.dx.resource.handlefrombinding(i32 2, i32 10, i32 1, i32 0, i1 false, ptr @Nine.str)

  ; RWBuffer<float4> Array[100] : register(u4, space3);
  ; RWBuffer<float4> B1 = Array[30];
  ; RWBuffer<float4> B2 = Array[42];
  ; resource array accesses should produce one metadata entry   
  %Array_30_h = call target("dx.TypedBuffer", <4 x float>, 1, 0, 0)
            @llvm.dx.resource.handlefrombinding(i32 3, i32 4, i32 100, i32 30, i1 false, ptr @Array.str)
  %Array_42_h = call target("dx.TypedBuffer", <4 x float>, 1, 0, 0)
            @llvm.dx.resource.handlefrombinding(i32 3, i32 4, i32 100, i32 42, i1 false, ptr @Array.str)

  ; test unbounded resource array
  ; RWBuffer<double> Array2[] : register(u2, space4);
  ; RWBuffer<double> C1 = Array[10];
  ; RWBuffer<double> C2 = Array[20];
  %Array2_10_h = call target("dx.TypedBuffer", double, 1, 0, 0)
            @llvm.dx.resource.handlefrombinding(i32 4, i32 2, i32 -1, i32 10, i1 false, ptr @Array2.str)
  %Array2_20_h = call target("dx.TypedBuffer", double, 1, 0, 0)
            @llvm.dx.resource.handlefrombinding(i32 4, i32 2, i32 -1, i32 20, i1 false, ptr @Array2.str)

  ; Same buffer type as Nine - should have the same type in metadata
  ; RWBuffer<double> Ten : register(u2);
  %Ten_h = call target("dx.TypedBuffer", i64, 1, 0, 0)
            @llvm.dx.resource.handlefrombinding(i32 5, i32 22, i32 1, i32 0, i1 false, ptr @Ten.str)

  ret void
}

attributes #0 = { noinline nounwind "hlsl.shader"="compute" }

; CHECK: %"RWBuffer<half4>" = type { <4 x half> }
; CHECK: %"RWBuffer<float2>" = type { <2 x float> }
; CHECK: %"RWBuffer<double>" = type { double }
; CHECK: %"RWBuffer<int32_t4>" = type { <4 x i32> }
; CHECK: %RWByteAddressBuffer = type { i32 }
; CHECK: %"RWStructuredBuffer<int16_t>" = type { i16 }
; CHECK: %"RasterizerOrderedBuffer<int32_t4>" = type { <4 x i32> }
; CHECK: %"RasterizerOrderedStructuredBuffer<int32_t4>" = type { <4 x i32> }
; CHECK: %RasterizerOrderedByteAddressBuffer = type { i32 }
; CHECK: %"RWBuffer<uint32_t>" = type { i64 }
; CHECK: %"RWBuffer<float4>" = type { <4 x float> }

; CHECK: @Zero = external constant %"RWBuffer<half4>"
; CHECK: @One = external constant %"RWBuffer<float2>"
; CHECK: @Two = external constant %"RWBuffer<double>"
; CHECK: @Three = external constant %"RWBuffer<int32_t4>"
; CHECK: @Four = external constant %RWByteAddressBuffer
; CHECK: @Five = external constant %"RWStructuredBuffer<int16_t>"
; CHECK: @Six = external constant %"RasterizerOrderedBuffer<int32_t4>"
; CHECK: @Seven = external constant %"RasterizerOrderedStructuredBuffer<int32_t4>"
; CHECK: @Eight = external constant %RasterizerOrderedByteAddressBuffer
; CHECK: @Nine = external constant %"RWBuffer<uint32_t>"
; CHECK: @Array = external constant [100 x %"RWBuffer<float4>"]
; CHECK: @Array2 = external constant [0 x %"RWBuffer<double>"]
; CHECK: @Ten = external constant %"RWBuffer<uint32_t>"

; CHECK: !dx.resources = !{[[ResList:[!][0-9]+]]}

; CHECK: [[ResList]] = !{null, [[UAVList:[!][0-9]+]], null, null}
; CHECK: [[UAVList]] = !{![[Zero:[0-9]+]], ![[One:[0-9]+]], ![[Two:[0-9]+]],
; CHECK-SAME: ![[Three:[0-9]+]], ![[Four:[0-9]+]], ![[Five:[0-9]+]],
; CHECK-SAME: ![[Six:[0-9]+]], ![[Seven:[0-9]+]], ![[Eight:[0-9]+]],
; CHECK-SAME: ![[Nine:[0-9]+]], ![[Array:[0-9]+]], ![[Array2:[0-9]+]], ![[Ten:[0-9]+]]}

; CHECK: ![[Zero]] = !{i32 0, ptr @Zero, !"Zero", i32 0, i32 0, i32 1, i32 10, i1 false, i1 false, i1 false, ![[Half:[0-9]+]]}
; CHECK: ![[Half]] = !{i32 0, i32 8}
; CHECK: ![[One]] = !{i32 1, ptr @One, !"One", i32 0, i32 1, i32 1, i32 10, i1 false, i1 false, i1 false, ![[Float:[0-9]+]]}
; CHECK: ![[Float]] = !{i32 0, i32 9}
; CHECK: ![[Two]] = !{i32 2, ptr @Two, !"Two", i32 0, i32 2, i32 1, i32 10, i1 false, i1 false, i1 false, ![[Double:[0-9]+]]}
; CHECK: ![[Double]] = !{i32 0, i32 10}
; CHECK: ![[Three]] = !{i32 3, ptr @Three, !"Three", i32 0, i32 3, i32 1, i32 10, i1 false, i1 false, i1 false, ![[I32:[0-9]+]]}
; CHECK: ![[I32]] = !{i32 0, i32 4}
; CHECK: ![[Four]] = !{i32 4, ptr @Four, !"Four", i32 0, i32 5, i32 1, i32 11, i1 false, i1 false, i1 false, null}
; CHECK: ![[Five]] = !{i32 5, ptr @Five, !"Five", i32 0, i32 6, i32 1, i32 12, i1 false, i1 false, i1 false, ![[FiveStride:[0-9]+]]}
; CHECK: ![[FiveStride]] = !{i32 1, i32 2}
; CHECK: ![[Six]] = !{i32 6, ptr @Six, !"Six", i32 0, i32 7, i32 1, i32 10, i1 false, i1 false, i1 true, ![[I32]]}
; CHECK: ![[Seven]] = !{i32 7, ptr @Seven, !"Seven", i32 0, i32 8, i32 1, i32 12, i1 false, i1 false, i1 true, ![[SevenStride:[0-9]+]]}
; CHECK: ![[SevenStride]] = !{i32 1, i32 16}
; CHECK: ![[Eight]] = !{i32 8, ptr @Eight, !"Eight", i32 0, i32 9, i32 1, i32 11, i1 false, i1 false, i1 true, null}
; CHECK: ![[Nine]] = !{i32 9, ptr @Nine, !"Nine", i32 2, i32 10, i32 1, i32 10, i1 false, i1 false, i1 false, ![[U64:[0-9]+]]}
; CHECK: ![[U64]] = !{i32 0, i32 7}
; CHECK: ![[Array]] = !{i32 10, ptr @Array, !"Array", i32 3, i32 4, i32 100, i32 10, i1 false, i1 false, i1 false, ![[Float]]}
; CHECK: ![[Array2]] = !{i32 11, ptr @Array2, !"Array2", i32 4, i32 2, i32 -1, i32 10, i1 false, i1 false, i1 false, ![[Double]]}
; CHECK: ![[Ten]] = !{i32 12, ptr @Ten, !"Ten", i32 5, i32 22, i32 1, i32 10, i1 false, i1 false, i1 false, ![[U64:[0-9]+]]}
