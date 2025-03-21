; RUN: opt -S -dxil-translate-metadata < %s | FileCheck %s
; RUN: opt -S --passes="dxil-pretty-printer" < %s 2>&1 | FileCheck %s --check-prefix=PRINT
; RUN: llc %s --filetype=asm -o - < %s 2>&1 | FileCheck %s --check-prefixes=CHECK,PRINT

target datalayout = "e-m:e-p:32:32-i1:32-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-n8:16:32:64"
target triple = "dxil-pc-shadermodel6.6-compute"

%"class.hlsl::Buffer" = type { target("dx.TypedBuffer", <4 x half>, 0, 0, 0) }
%"class.hlsl::Buffer.1" = type { target("dx.TypedBuffer", <2 x float>, 0, 0, 0) }
%"class.hlsl::Buffer.2" = type { target("dx.TypedBuffer", double, 0, 0, 0) }
%"class.hlsl::Buffer.3" = type { target("dx.TypedBuffer", i32, 0, 0, 1) }
%"class.hlsl::ByteAddressBuffer" = type { target("dx.RawBuffer", i8, 0, 0) }
%"class.hlsl::StructuredBuffer" = type { target("dx.RawBuffer", i16, 0, 0) }
%"class.hlsl::Buffer.4" = type { target("dx.TypedBuffer", i64, 0, 0, 0) }

@Zero = internal global %"class.hlsl::Buffer" poison, align 4
@One = internal global %"class.hlsl::Buffer.1" poison, align 4
@Two = internal global %"class.hlsl::Buffer.2" poison, align 4
@Three = internal global %"class.hlsl::Buffer.3" poison, align 4
@Four = internal global %"class.hlsl::ByteAddressBuffer" poison, align 4
@Five = internal global %"class.hlsl::StructuredBuffer" poison, align 4
@Six = internal global %"class.hlsl::Buffer.4" poison, align 4

; PRINT:; Resource Bindings:
; PRINT-NEXT:;
; PRINT-NEXT:; Name                                 Type  Format         Dim      ID      HLSL Bind  Count
; PRINT-NEXT:; ------------------------------ ---------- ------- ----------- ------- -------------- ------
; PRINT-NEXT:;                                   texture     f16         buf      T0             t0     1
; PRINT-NEXT:;                                   texture     f32         buf      T1             t1     1
; PRINT-NEXT:;                                   texture     f64         buf      T2             t2     1
; PRINT-NEXT:;                                   texture     i32         buf      T3             t3     1
; PRINT-NEXT:;                                   texture    byte         r/o      T4             t5     1
; PRINT-NEXT:;                                   texture  struct         r/o      T5             t6     1
; PRINT-NEXT:;                                   texture     u64         buf      T6     t10,space2     1
; PRINT-NEXT:;                                   texture     f32         buf      T7      t4,space3   100

define void @test() #0 {
  ; Buffer<half4> Buf : register(t0)
  %Zero_h = call target("dx.TypedBuffer", <4 x half>, 0, 0, 0)
            @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, i1 false)
  store target("dx.TypedBuffer", <4 x half>, 0, 0, 0) %Zero_h, ptr @Zero, align 4
 
  ; Buffer<float4> Buf : register(t1)
  %One_h = call target("dx.TypedBuffer", <2 x float>, 0, 0, 0)
            @llvm.dx.resource.handlefrombinding(i32 0, i32 1, i32 1, i32 0, i1 false)
  store target("dx.TypedBuffer", <2 x float>, 0, 0, 0) %One_h, ptr @One, align 4
 
  ; Buffer<double> Two : register(t2);
  %Two_h = call target("dx.TypedBuffer", double, 0, 0, 0)
            @llvm.dx.resource.handlefrombinding(i32 0, i32 2, i32 1, i32 0, i1 false)
  store target("dx.TypedBuffer", double, 0, 0, 0) %Two_h, ptr @Two, align 4

  ; Buffer<int4> Three : register(t3);
  %Three_h = call target("dx.TypedBuffer", <4 x i32>, 0, 0, 1)
            @llvm.dx.resource.handlefrombinding(i32 0, i32 3, i32 1, i32 0, i1 false)
  store target("dx.TypedBuffer", <4 x i32>, 0, 0, 1) %Three_h, ptr @Three, align 4

  ; ByteAddressBuffer Four : register(t4)
  %Four_h = call target("dx.RawBuffer", i8, 0, 0)
            @llvm.dx.resource.handlefrombinding(i32 0, i32 5, i32 1, i32 0, i1 false)
  store target("dx.RawBuffer", i8, 0, 0) %Four_h, ptr @Four, align 4

  ; StructuredBuffer<int16_t> Five : register(t6);
  %Five_h = call target("dx.RawBuffer", i16, 0, 0)
            @llvm.dx.resource.handlefrombinding(i32 0, i32 6, i32 1, i32 0, i1 false)
  store target("dx.RawBuffer", i16, 0, 0) %Five_h, ptr @Five, align 4  
  
  ; Buffer<double> Six : register(t10, space2);
  %Six_h = call target("dx.TypedBuffer", i64, 0, 0, 0)
            @llvm.dx.resource.handlefrombinding(i32 2, i32 10, i32 1, i32 0, i1 false)
  store target("dx.TypedBuffer", i64, 0, 0, 0) %Six_h, ptr @Six, align 4

  ; Buffer<float4> Array[100] : register(t4, space3);
  ; Buffer<float4> B1 = Array[30];
  ; Buffer<float4> B1 = Array[42];
  ; resource array accesses should produce one metadata entry   
  %Array_30_h = call target("dx.TypedBuffer", <4 x float>, 0, 0, 0)
            @llvm.dx.resource.handlefrombinding(i32 3, i32 4, i32 100, i32 30, i1 false)
  %Array_42_h = call target("dx.TypedBuffer", <4 x float>, 0, 0, 0)
            @llvm.dx.resource.handlefrombinding(i32 3, i32 4, i32 100, i32 42, i1 false)

  ret void
}

attributes #0 = { noinline nounwind "hlsl.shader"="compute" }

; CHECK: !dx.resources = !{[[ResList:[!][0-9]+]]}

; CHECK: [[ResList]] = !{[[SRVList:[!][0-9]+]], null, null, null}
; CHECK: [[SRVList]] = !{![[Zero:[0-9]+]], ![[One:[0-9]+]], ![[Two:[0-9]+]],
; CHECK-SAME: ![[Three:[0-9]+]], ![[Four:[0-9]+]], ![[Five:[0-9]+]],
; CHECK-SAME: ![[Six:[0-9]+]], ![[Array:[0-9]+]]}

; CHECK: ![[Zero]] = !{i32 0, ptr @0, !"", i32 0, i32 0, i32 1, i32 10, i32 0, ![[Half:[0-9]+]]}
; CHECK: ![[Half]] = !{i32 0, i32 8}
; CHECK: ![[One]] = !{i32 1, ptr @1, !"", i32 0, i32 1, i32 1, i32 10, i32 0, ![[Float:[0-9]+]]}
; CHECK: ![[Float]] = !{i32 0, i32 9}
; CHECK: ![[Two]] = !{i32 2, ptr @2, !"", i32 0, i32 2, i32 1, i32 10, i32 0, ![[Double:[0-9]+]]}
; CHECK: ![[Double]] = !{i32 0, i32 10}
; CHECK: ![[Three]] = !{i32 3, ptr @3, !"", i32 0, i32 3, i32 1, i32 10, i32 0, ![[I32:[0-9]+]]}
; CHECK: ![[I32]] = !{i32 0, i32 4}
; CHECK: ![[Four]] = !{i32 4, ptr @4, !"", i32 0, i32 5, i32 1, i32 11, i32 0, null}
; CHECK: ![[Five]] = !{i32 5, ptr @5, !"", i32 0, i32 6, i32 1, i32 12, i32 0, ![[FiveStride:[0-9]+]]}
; CHECK: ![[FiveStride]] = !{i32 1, i32 2}
; CHECK: ![[Six]] = !{i32 6, ptr @6, !"", i32 2, i32 10, i32 1, i32 10, i32 0, ![[U64:[0-9]+]]}
; CHECK: ![[U64]] = !{i32 0, i32 7}
; CHECK: ![[Array]] = !{i32 7, ptr @7, !"", i32 3, i32 4, i32 100, i32 10, i32 0, ![[Float]]}
