; RUN: llc -mtriple=x86_64-pc-linux -mattr=mmx < %s | FileCheck %s

; MMX packed sub opcodes were wrongly marked as commutative.
; This test checks that the operands of packed sub instructions are
; never interchanged by the "Two-Address instruction pass".

declare { i64, double } @getFirstParam()
declare { i64, double } @getSecondParam()

define i64 @test_psubb() {
entry:
  %call = tail call { i64, double } @getFirstParam()
  %0 = extractvalue { i64, double } %call, 0
  %call2 = tail call { i64, double } @getSecondParam()
  %1 = extractvalue { i64, double } %call2, 0
  %__m1.0.insert.i = insertelement <1 x i64> undef, i64 %0, i32 0
  %__m2.0.insert.i = insertelement <1 x i64> undef, i64 %1, i32 0
  %2 = bitcast <1 x i64> %__m1.0.insert.i to <8 x i8>
  %3 = bitcast <8 x i8> %2 to x86_mmx
  %4 = bitcast <1 x i64> %__m2.0.insert.i to <8 x i8>
  %5 = bitcast <8 x i8> %4 to x86_mmx
  %6 = tail call x86_mmx @llvm.x86.mmx.psub.b(x86_mmx %3, x86_mmx %5) nounwind
  %7 = bitcast x86_mmx %6 to <8 x i8>
  %8 = bitcast <8 x i8> %7 to <1 x i64>
  %retval.0.extract.i15 = extractelement <1 x i64> %8, i32 0
  ret i64 %retval.0.extract.i15
}

; CHECK-LABEL: test_psubb:
; CHECK:   callq getFirstParam
; CHECK:   movq %rax, [[TEMP:%[a-z0-9]+]]
; CHECK:   callq getSecondParam
; CHECK:   movq [[TEMP]], [[PARAM1:%[a-z0-9]+]]
; CHECK:   movq %rax, [[PARAM2:%[a-z0-9]+]]
; CHECK:   psubb [[PARAM2]], [[PARAM1]]
; CHECK: ret

define i64 @test_psubw() {
entry:
  %call = tail call { i64, double } @getFirstParam()
  %0 = extractvalue { i64, double } %call, 0
  %call2 = tail call { i64, double } @getSecondParam()
  %1 = extractvalue { i64, double } %call2, 0
  %__m1.0.insert.i = insertelement <1 x i64> undef, i64 %0, i32 0
  %__m2.0.insert.i = insertelement <1 x i64> undef, i64 %1, i32 0
  %2 = bitcast <1 x i64> %__m1.0.insert.i to <4 x i16>
  %3 = bitcast <4 x i16> %2 to x86_mmx
  %4 = bitcast <1 x i64> %__m2.0.insert.i to <4 x i16>
  %5 = bitcast <4 x i16> %4 to x86_mmx
  %6 = tail call x86_mmx @llvm.x86.mmx.psub.w(x86_mmx %3, x86_mmx %5) nounwind
  %7 = bitcast x86_mmx %6 to <4 x i16>
  %8 = bitcast <4 x i16> %7 to <1 x i64>
  %retval.0.extract.i15 = extractelement <1 x i64> %8, i32 0
  ret i64 %retval.0.extract.i15
}

; CHECK-LABEL: test_psubw:
; CHECK:   callq getFirstParam
; CHECK:   movq %rax, [[TEMP:%[a-z0-9]+]]
; CHECK:   callq getSecondParam
; CHECK:   movq [[TEMP]], [[PARAM1:%[a-z0-9]+]]
; CHECK:   movq %rax, [[PARAM2:%[a-z0-9]+]]
; CHECK:   psubw [[PARAM2]], [[PARAM1]]
; CHECK: ret


define i64 @test_psubd() {
entry:
  %call = tail call { i64, double } @getFirstParam()
  %0 = extractvalue { i64, double } %call, 0
  %call2 = tail call { i64, double } @getSecondParam()
  %1 = extractvalue { i64, double } %call2, 0
  %__m1.0.insert.i = insertelement <1 x i64> undef, i64 %0, i32 0
  %__m2.0.insert.i = insertelement <1 x i64> undef, i64 %1, i32 0
  %2 = bitcast <1 x i64> %__m1.0.insert.i to <2 x i32>
  %3 = bitcast <2 x i32> %2 to x86_mmx
  %4 = bitcast <1 x i64> %__m2.0.insert.i to <2 x i32>
  %5 = bitcast <2 x i32> %4 to x86_mmx
  %6 = tail call x86_mmx @llvm.x86.mmx.psub.d(x86_mmx %3, x86_mmx %5) nounwind
  %7 = bitcast x86_mmx %6 to <2 x i32>
  %8 = bitcast <2 x i32> %7 to <1 x i64>
  %retval.0.extract.i15 = extractelement <1 x i64> %8, i32 0
  ret i64 %retval.0.extract.i15
}

; CHECK-LABEL: test_psubd:
; CHECK:   callq getFirstParam
; CHECK:   movq %rax, [[TEMP:%[a-z0-9]+]]
; CHECK:   callq getSecondParam
; CHECK:   movq [[TEMP]], [[PARAM1:%[a-z0-9]+]]
; CHECK:   movq %rax, [[PARAM2:%[a-z0-9]+]]
; CHECK:   psubd [[PARAM2]], [[PARAM1]]
; CHECK: ret

define i64 @test_psubsb() {
entry:
  %call = tail call { i64, double } @getFirstParam()
  %0 = extractvalue { i64, double } %call, 0
  %call2 = tail call { i64, double } @getSecondParam()
  %1 = extractvalue { i64, double } %call2, 0
  %__m1.0.insert.i = insertelement <1 x i64> undef, i64 %0, i32 0
  %__m2.0.insert.i = insertelement <1 x i64> undef, i64 %1, i32 0
  %2 = bitcast <1 x i64> %__m1.0.insert.i to <8 x i8>
  %3 = bitcast <8 x i8> %2 to x86_mmx
  %4 = bitcast <1 x i64> %__m2.0.insert.i to <8 x i8>
  %5 = bitcast <8 x i8> %4 to x86_mmx
  %6 = tail call x86_mmx @llvm.x86.mmx.psubs.b(x86_mmx %3, x86_mmx %5) nounwind
  %7 = bitcast x86_mmx %6 to <8 x i8>
  %8 = bitcast <8 x i8> %7 to <1 x i64>
  %retval.0.extract.i15 = extractelement <1 x i64> %8, i32 0
  ret i64 %retval.0.extract.i15
}

; CHECK-LABEL: test_psubsb:
; CHECK:   callq getFirstParam
; CHECK:   movq %rax, [[TEMP:%[a-z0-9]+]]
; CHECK:   callq getSecondParam
; CHECK:   movq [[TEMP]], [[PARAM1:%[a-z0-9]+]]
; CHECK:   movq %rax, [[PARAM2:%[a-z0-9]+]]
; CHECK:   psubsb [[PARAM2]], [[PARAM1]]
; CHECK: ret

define i64 @test_psubswv() {
entry:
  %call = tail call { i64, double } @getFirstParam()
  %0 = extractvalue { i64, double } %call, 0
  %call2 = tail call { i64, double } @getSecondParam()
  %1 = extractvalue { i64, double } %call2, 0
  %__m1.0.insert.i = insertelement <1 x i64> undef, i64 %0, i32 0
  %__m2.0.insert.i = insertelement <1 x i64> undef, i64 %1, i32 0
  %2 = bitcast <1 x i64> %__m1.0.insert.i to <4 x i16>
  %3 = bitcast <4 x i16> %2 to x86_mmx
  %4 = bitcast <1 x i64> %__m2.0.insert.i to <4 x i16>
  %5 = bitcast <4 x i16> %4 to x86_mmx
  %6 = tail call x86_mmx @llvm.x86.mmx.psubs.w(x86_mmx %3, x86_mmx %5) nounwind
  %7 = bitcast x86_mmx %6 to <4 x i16>
  %8 = bitcast <4 x i16> %7 to <1 x i64>
  %retval.0.extract.i15 = extractelement <1 x i64> %8, i32 0
  ret i64 %retval.0.extract.i15
}

; CHECK-LABEL: test_psubswv:
; CHECK:   callq getFirstParam
; CHECK:   movq %rax, [[TEMP:%[a-z0-9]+]]
; CHECK:   callq getSecondParam
; CHECK:   movq [[TEMP]], [[PARAM1:%[a-z0-9]+]]
; CHECK:   movq %rax, [[PARAM2:%[a-z0-9]+]]
; CHECK:   psubsw [[PARAM2]], [[PARAM1]]
; CHECK: ret

define i64 @test_psubusbv() {
entry:
  %call = tail call { i64, double } @getFirstParam()
  %0 = extractvalue { i64, double } %call, 0
  %call2 = tail call { i64, double } @getSecondParam()
  %1 = extractvalue { i64, double } %call2, 0
  %__m1.0.insert.i = insertelement <1 x i64> undef, i64 %0, i32 0
  %__m2.0.insert.i = insertelement <1 x i64> undef, i64 %1, i32 0
  %2 = bitcast <1 x i64> %__m1.0.insert.i to <8 x i8>
  %3 = bitcast <8 x i8> %2 to x86_mmx
  %4 = bitcast <1 x i64> %__m2.0.insert.i to <8 x i8>
  %5 = bitcast <8 x i8> %4 to x86_mmx
  %6 = tail call x86_mmx @llvm.x86.mmx.psubus.b(x86_mmx %3, x86_mmx %5) nounwind
  %7 = bitcast x86_mmx %6 to <8 x i8>
  %8 = bitcast <8 x i8> %7 to <1 x i64>
  %retval.0.extract.i15 = extractelement <1 x i64> %8, i32 0
  ret i64 %retval.0.extract.i15
}

; CHECK-LABEL: test_psubusbv:
; CHECK:   callq getFirstParam
; CHECK:   movq %rax, [[TEMP:%[a-z0-9]+]]
; CHECK:   callq getSecondParam
; CHECK:   movq [[TEMP]], [[PARAM1:%[a-z0-9]+]]
; CHECK:   movq %rax, [[PARAM2:%[a-z0-9]+]]
; CHECK:   psubusb [[PARAM2]], [[PARAM1]]
; CHECK: ret

define i64 @test_psubuswv() {
entry:
  %call = tail call { i64, double } @getFirstParam()
  %0 = extractvalue { i64, double } %call, 0
  %call2 = tail call { i64, double } @getSecondParam()
  %1 = extractvalue { i64, double } %call2, 0
  %__m1.0.insert.i = insertelement <1 x i64> undef, i64 %0, i32 0
  %__m2.0.insert.i = insertelement <1 x i64> undef, i64 %1, i32 0
  %2 = bitcast <1 x i64> %__m1.0.insert.i to <4 x i16>
  %3 = bitcast <4 x i16> %2 to x86_mmx
  %4 = bitcast <1 x i64> %__m2.0.insert.i to <4 x i16>
  %5 = bitcast <4 x i16> %4 to x86_mmx
  %6 = tail call x86_mmx @llvm.x86.mmx.psubus.w(x86_mmx %3, x86_mmx %5) nounwind
  %7 = bitcast x86_mmx %6 to <4 x i16>
  %8 = bitcast <4 x i16> %7 to <1 x i64>
  %retval.0.extract.i15 = extractelement <1 x i64> %8, i32 0
  ret i64 %retval.0.extract.i15
}

; CHECK-LABEL: test_psubuswv:
; CHECK:   callq getFirstParam
; CHECK:   movq %rax, [[TEMP:%[a-z0-9]+]]
; CHECK:   callq getSecondParam
; CHECK:   movq [[TEMP]], [[PARAM1:%[a-z0-9]+]]
; CHECK:   movq %rax, [[PARAM2:%[a-z0-9]+]]
; CHECK:   psubusw [[PARAM2]], [[PARAM1]]
; CHECK: ret


declare x86_mmx @llvm.x86.mmx.psubus.w(x86_mmx, x86_mmx) nounwind readnone

declare x86_mmx @llvm.x86.mmx.psubus.b(x86_mmx, x86_mmx) nounwind readnone

declare x86_mmx @llvm.x86.mmx.psubs.w(x86_mmx, x86_mmx) nounwind readnone

declare x86_mmx @llvm.x86.mmx.psubs.b(x86_mmx, x86_mmx) nounwind readnone

declare x86_mmx @llvm.x86.mmx.psub.d(x86_mmx, x86_mmx) nounwind readnone

declare x86_mmx @llvm.x86.mmx.psub.w(x86_mmx, x86_mmx) nounwind readnone

declare x86_mmx @llvm.x86.mmx.psub.b(x86_mmx, x86_mmx) nounwind readnone
