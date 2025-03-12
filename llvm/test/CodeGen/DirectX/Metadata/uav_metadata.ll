; RUN: opt -S -dxil-translate-metadata < %s | FileCheck %s
; RUN: opt -S --passes="dxil-pretty-printer" < %s 2>&1 | FileCheck %s --check-prefix=PRINT
; RUN: llc %s --filetype=asm -o - < %s 2>&1 | FileCheck %s --check-prefixes=CHECK,PRINT

target datalayout = "e-m:e-p:32:32-i1:32-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-n8:16:32:64"
target triple = "dxil-pc-shadermodel6.6-compute"

%"class.hlsl::RWBuffer" = type { target("dx.TypedBuffer", <4 x half>, 1, 0, 0) }
%"class.hlsl::RWBuffer.1" = type { target("dx.TypedBuffer", <2 x float>, 1, 0, 0) }
%"class.hlsl::RWBuffer.2" = type { target("dx.TypedBuffer", double, 1, 0, 0) }
%"class.hlsl::RWBuffer.3" = type { target("dx.TypedBuffer", i32, 1, 0, 1) }
%"class.hlsl::RWByteAddressBuffer" = type { target("dx.RawBuffer", i8, 1, 0) }
%"class.hlsl::RWStructuredBuffer" = type { target("dx.RawBuffer", i16, 1, 0) }
%"class.hlsl::RasterizerOrderedBuffer" = type { target("dx.TypedBuffer", <4 x i32>, 1, 1, 1) }
%"class.hlsl::RasterizerOrderedStructuredBuffer" = type { target("dx.RawBuffer", <4 x i32>, 1, 1) }
%"class.hlsl::RasterizerOrderedByteAddressBuffer" = type { target("dx.RawBuffer", i8, 1, 1) }
%"class.hlsl::RWBuffer.4" = type { target("dx.TypedBuffer", i64, 1, 0, 0) }

@Zero = internal global %"class.hlsl::RWBuffer" poison, align 4
@One = internal global %"class.hlsl::RWBuffer.1" poison, align 4
@Two = internal global %"class.hlsl::RWBuffer.2" poison, align 4
@Three = internal global %"class.hlsl::RWBuffer.3" poison, align 4
@Four = internal global %"class.hlsl::RWByteAddressBuffer" poison, align 4
@Five = internal global %"class.hlsl::RWStructuredBuffer" poison, align 4
@Six = internal global %"class.hlsl::RasterizerOrderedBuffer" poison, align 4
@Seven = internal global %"class.hlsl::RasterizerOrderedStructuredBuffer" poison, align 4
@Eight = internal global %"class.hlsl::RasterizerOrderedByteAddressBuffer" poison, align 4
@Nine = internal global %"class.hlsl::RWBuffer.4" poison, align 4

; PRINT:; Resource Bindings:
; PRINT-NEXT:;
; PRINT-NEXT:; Name                                 Type  Format         Dim      ID      HLSL Bind  Count
; PRINT-NEXT:; ------------------------------ ---------- ------- ----------- ------- -------------- ------
; PRINT-NEXT:;                                       UAV     f16         buf      U0             u0     1
; PRINT-NEXT:;                                       UAV     f32         buf      U1             u1     1
; PRINT-NEXT:;                                       UAV     f64         buf      U2             u2     1
; PRINT-NEXT:;                                       UAV     i32         buf      U3             u3     1
; PRINT-NEXT:;                                       UAV    byte         r/w      U4             u5     1
; PRINT-NEXT:;                                       UAV  struct         r/w      U5             u6     1
; PRINT-NEXT:;                                       UAV     i32         buf      U6             u7     1
; PRINT-NEXT:;                                       UAV  struct         r/w      U7             u8     1
; PRINT-NEXT:;                                       UAV    byte         r/w      U8             u9     1
; PRINT-NEXT:;                                       UAV     u64         buf      U9     u10,space2     1
; PRINT-NEXT:;                                       UAV     f32         buf     U10      u4,space3   100

define void @test() #0 {
  ; RWBuffer<half4> Zero : register(u0)
  %Zero_h = call target("dx.TypedBuffer", <4 x half>, 1, 0, 0)
            @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, i1 false)
  store target("dx.TypedBuffer", <4 x half>, 1, 0, 0) %Zero_h, ptr @Zero, align 4
 
  ; RWBuffer<float4> One : register(u1)
  %One_h = call target("dx.TypedBuffer", <2 x float>, 1, 0, 0)
            @llvm.dx.resource.handlefrombinding(i32 0, i32 1, i32 1, i32 0, i1 false)
  store target("dx.TypedBuffer", <2 x float>, 1, 0, 0) %One_h, ptr @One, align 4
 
  ; RWBuffer<double> Two : register(u2);
  %Two_h = call target("dx.TypedBuffer", double, 1, 0, 0)
            @llvm.dx.resource.handlefrombinding(i32 0, i32 2, i32 1, i32 0, i1 false)
  store target("dx.TypedBuffer", double, 1, 0, 0) %Two_h, ptr @Two, align 4

  ; RWBuffer<int4> Three : register(u3);
  %Three_h = call target("dx.TypedBuffer", <4 x i32>, 1, 0, 1)
            @llvm.dx.resource.handlefrombinding(i32 0, i32 3, i32 1, i32 0, i1 false)
  store target("dx.TypedBuffer", <4 x i32>, 1, 0, 1) %Three_h, ptr @Three, align 4

  ; ByteAddressBuffer Four : register(u5)
  %Four_h = call target("dx.RawBuffer", i8, 1, 0)
            @llvm.dx.resource.handlefrombinding(i32 0, i32 5, i32 1, i32 0, i1 false)
  store target("dx.RawBuffer", i8, 1, 0) %Four_h, ptr @Four, align 4

  ; RWStructuredBuffer<int16_t> Five : register(u6);
  %Five_h = call target("dx.RawBuffer", i16, 1, 0)
            @llvm.dx.resource.handlefrombinding(i32 0, i32 6, i32 1, i32 0, i1 false)
  store target("dx.RawBuffer", i16, 1, 0) %Five_h, ptr @Five, align 4
  
  ; RasterizerOrderedBuffer<int4> Six : register(u7);
  %Six_h = call target("dx.TypedBuffer", <4 x i32>, 1, 1, 1)
            @llvm.dx.resource.handlefrombinding(i32 0, i32 7, i32 1, i32 0, i1 false)
  store target("dx.TypedBuffer", <4 x i32>, 1, 1, 1) %Six_h, ptr @Six, align 4

  ; RasterizerOrderedStructuredBuffer<uint4> Seven : register(u3, space10);
  %Seven_h = call target("dx.RawBuffer", <4 x i32>, 1, 1)
            @llvm.dx.resource.handlefrombinding(i32 0, i32 8, i32 1, i32 0, i1 false)
  store target("dx.RawBuffer", <4 x i32>, 1, 1) %Seven_h, ptr @Seven, align 4

  ; RasterizerOrderedByteAddressBuffer Eight : register(u9); 
  %Eight_h = call target("dx.RawBuffer", i8, 1, 1) 
            @llvm.dx.resource.handlefrombinding(i32 0, i32 9, i32 1, i32 0, i1 false)
  store target("dx.RawBuffer", i8, 1, 1) %Eight_h, ptr @Eight, align 4

  ; RWBuffer<double> Nine : register(u2);
  %Nine_h = call target("dx.TypedBuffer", i64, 1, 0, 0)
            @llvm.dx.resource.handlefrombinding(i32 2, i32 10, i32 1, i32 0, i1 false)
  store target("dx.TypedBuffer", i64, 1, 0, 0) %Nine_h, ptr @Nine, align 4

  ; RWBuffer<float4> Array[100] : register(u4, space3);
  ; RWBuffer<float4> B1 = Array[30];
  ; RWBuffer<float4> B1 = Array[42];
  ; resource array accesses should produce one metadata entry   
  %Array_30_h = call target("dx.TypedBuffer", <4 x float>, 1, 0, 0)
            @llvm.dx.resource.handlefrombinding(i32 3, i32 4, i32 100, i32 30, i1 false)
  %Array_42_h = call target("dx.TypedBuffer", <4 x float>, 1, 0, 0)
            @llvm.dx.resource.handlefrombinding(i32 3, i32 4, i32 100, i32 42, i1 false)

  ret void
}

attributes #0 = { noinline nounwind "hlsl.shader"="compute" }

; CHECK: !dx.resources = !{[[ResList:[!][0-9]+]]}

; CHECK: [[ResList]] = !{null, [[UAVList:[!][0-9]+]], null, null}
; CHECK: [[UAVList]] = !{![[Zero:[0-9]+]], ![[One:[0-9]+]], ![[Two:[0-9]+]],
; CHECK-SAME: ![[Three:[0-9]+]], ![[Four:[0-9]+]], ![[Five:[0-9]+]],
; CHECK-SAME: ![[Six:[0-9]+]], ![[Seven:[0-9]+]], ![[Eight:[0-9]+]],
; CHECK-SAME: ![[Nine:[0-9]+]], ![[Array:[0-9]+]]}

; CHECK: ![[Zero]] = !{i32 0, ptr @0, !"", i32 0, i32 0, i32 1, i32 10, i1 false, i1 false, i1 false, ![[Half:[0-9]+]]}
; CHECK: ![[Half]] = !{i32 0, i32 8}
; CHECK: ![[One]] = !{i32 1, ptr @1, !"", i32 0, i32 1, i32 1, i32 10, i1 false, i1 false, i1 false, ![[Float:[0-9]+]]}
; CHECK: ![[Float]] = !{i32 0, i32 9}
; CHECK: ![[Two]] = !{i32 2, ptr @2, !"", i32 0, i32 2, i32 1, i32 10, i1 false, i1 false, i1 false, ![[Double:[0-9]+]]}
; CHECK: ![[Double]] = !{i32 0, i32 10}
; CHECK: ![[Three]] = !{i32 3, ptr @3, !"", i32 0, i32 3, i32 1, i32 10, i1 false, i1 false, i1 false, ![[I32:[0-9]+]]}
; CHECK: ![[I32]] = !{i32 0, i32 4}
; CHECK: ![[Four]] = !{i32 4, ptr @4, !"", i32 0, i32 5, i32 1, i32 11, i1 false, i1 false, i1 false, null}
; CHECK: ![[Five]] = !{i32 5, ptr @5, !"", i32 0, i32 6, i32 1, i32 12, i1 false, i1 false, i1 false, ![[FiveStride:[0-9]+]]}
; CHECK: ![[FiveStride]] = !{i32 1, i32 2}
; CHECK: ![[Six]] = !{i32 6, ptr @6, !"", i32 0, i32 7, i32 1, i32 10, i1 false, i1 false, i1 true, ![[I32]]}
; CHECK: ![[Seven]] = !{i32 7, ptr @7, !"", i32 0, i32 8, i32 1, i32 12, i1 false, i1 false, i1 true, ![[SevenStride:[0-9]+]]}
; CHECK: ![[SevenStride]] = !{i32 1, i32 16}
; CHECK: ![[Eight]] = !{i32 8, ptr @8, !"", i32 0, i32 9, i32 1, i32 11, i1 false, i1 false, i1 true, null}
; CHECK: ![[Nine]] = !{i32 9, ptr @9, !"", i32 2, i32 10, i32 1, i32 10, i1 false, i1 false, i1 false, ![[U64:[0-9]+]]}
; CHECK: ![[U64]] = !{i32 0, i32 7}
; CHECK: ![[Array]] = !{i32 10, ptr @10, !"", i32 3, i32 4, i32 100, i32 10, i1 false, i1 false, i1 false, ![[Float]]}
