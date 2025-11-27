; RUN: opt -S -dxil-translate-metadata < %s | FileCheck %s
; RUN: opt -S --passes="dxil-pretty-printer" < %s 2>&1 | FileCheck %s --check-prefix=PRINT
; RUN: llc %s --filetype=asm -o - < %s 2>&1 | FileCheck %s --check-prefixes=CHECK,PRINT

target triple = "dxil-pc-shadermodel6.6-compute"

%__cblayout_CB1 = type <{ float, i32, double, <2 x i32> }>
@CB1.cb = global target("dx.CBuffer", %__cblayout_CB1) poison
@CB1.str = private unnamed_addr constant [4 x i8] c"CB1\00", align 1

%__cblayout_CB2 = type <{ float, target("dx.Padding", 4), double, float, half, i16, i64, i32 }>
@CB2.cb = global target("dx.CBuffer", %__cblayout_CB2) poison
@CB2.str = private unnamed_addr constant [4 x i8] c"CB2\00", align 1

%__cblayout_MyConstants = type <{
  double, target("dx.Padding", 8),
  <3 x float>, float,
  <3 x double>, half, target("dx.Padding", 6),
  <2 x double>,
  float, <3 x half>, <3 x half>
}>
@MyConstants.cb = global target("dx.CBuffer", %__cblayout_MyConstants) poison
@MyConstants.str = private unnamed_addr constant [12 x i8] c"MyConstants\00", align 1

; PRINT:; Resource Bindings:
; PRINT-NEXT:;
; PRINT-NEXT:; Name            Type  Format  Dim   ID    HLSL Bind  Count
; PRINT-NEXT:; ----
; PRINT-NEXT:; CB1          cbuffer      NA   NA  CB0          cb0     1
; PRINT-NEXT:; CB2          cbuffer      NA   NA  CB1          cb1     1
; PRINT-NEXT:; MyConstants  cbuffer      NA   NA  CB2  cb5,space15     1

define void @test() #0 {

  ; cbuffer CB1 : register(b0) {
  ;   float a;
  ;   int b;
  ;   double c;
  ;   int2 d;
  ; }
  %CB1.cb_h = call target("dx.CBuffer", %__cblayout_CB1)
            @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr @CB1.str)

  ; cbuffer CB2 : register(b0) {
  ;   float a;
  ;   double b;
  ;   float c;
  ;   half d;
  ;   uint16_t e;
  ;   int64_t f;
  ;   int g;
  ;}
  %CB2.cb_h = call target("dx.CBuffer", %__cblayout_CB2)
            @llvm.dx.resource.handlefrombinding(i32 0, i32 1, i32 1, i32 0, ptr @CB2.str)

  ; cbuffer CB3 : register(b5) {
  ;   double B0;
  ;   float3 B1;
  ;   float B2;
  ;   double3 B3;
  ;   half B4;
  ;   double2 B5;
  ;   float B6;
  ;   half3 B7;
  ;   half3 B8;
  ; }
  %CB3.cb_h = call target("dx.CBuffer", %__cblayout_MyConstants)
            @llvm.dx.resource.handlefrombinding(i32 15, i32 5, i32 1, i32 0, ptr @MyConstants.str)

  ret void
}

attributes #0 = { noinline nounwind "hlsl.shader"="compute" }

; CHECK: %CBuffer.CB1 = type { { float, i32, double, <2 x i32> } }
; CHECK: %CBuffer.CB2 = type { { float, double, float, half, i16, i64, i32 } }
; CHECK: %CBuffer.MyConstants = type { { double, <3 x float>, float, <3 x double>, half, <2 x double>, float, <3 x half>, <3 x half> } }

; CHECK: @CB1 = external constant %CBuffer.CB1
; CHECK: @CB2 = external constant %CBuffer.CB2
; CHECK: @MyConstants = external constant %CBuffer.MyConstants

; CHECK: !dx.resources = !{[[ResList:[!][0-9]+]]}

; CHECK: [[ResList]] = !{null, null, [[CBList:[!][0-9]+]], null}
; CHECK: [[CBList]] = !{![[CB1:[0-9]+]], ![[CB2:[0-9]+]], ![[MYCONSTANTS:[0-9]+]]}
; CHECK: ![[CB1]] = !{i32 0, ptr @CB1, !"CB1", i32 0, i32 0, i32 1, i32 24, null}
; CHECK: ![[CB2]] = !{i32 1, ptr @CB2, !"CB2", i32 0, i32 1, i32 1, i32 36, null}
; CHECK: ![[MYCONSTANTS]] = !{i32 2, ptr @MyConstants, !"MyConstants", i32 15, i32 5, i32 1, i32 96, null}
