; Test loading of 64-bit constants.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare void @foo(i64, i64, i64, i64)
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture, i64, i1 immarg)

; Check 0.
define i64 @f1() {
; CHECK-LABEL: f1:
; CHECK: lghi %r2, 0
; CHECK-NEXT: br %r14
  ret i64 0
}

; Check the high end of the LGHI range.
define i64 @f2() {
; CHECK-LABEL: f2:
; CHECK: lghi %r2, 32767
; CHECK-NEXT: br %r14
  ret i64 32767
}

; Check the next value up, which must use LLILL instead.
define i64 @f3() {
; CHECK-LABEL: f3:
; CHECK: llill %r2, 32768
; CHECK-NEXT: br %r14
  ret i64 32768
}

; Check the high end of the LLILL range.
define i64 @f4() {
; CHECK-LABEL: f4:
; CHECK: llill %r2, 65535
; CHECK-NEXT: br %r14
  ret i64 65535
}

; Check the first useful LLILH value, which is the next one up.
define i64 @f5() {
; CHECK-LABEL: f5:
; CHECK: llilh %r2, 1
; CHECK-NEXT: br %r14
  ret i64 65536
}

; Check the first useful LGFI value, which is the next one up again.
define i64 @f6() {
; CHECK-LABEL: f6:
; CHECK: lgfi %r2, 65537
; CHECK-NEXT: br %r14
  ret i64 65537
}

; Check the high end of the LGFI range.
define i64 @f7() {
; CHECK-LABEL: f7:
; CHECK: lgfi %r2, 2147483647
; CHECK-NEXT: br %r14
  ret i64 2147483647
}

; Check the next value up, which should use LLILH instead.
define i64 @f8() {
; CHECK-LABEL: f8:
; CHECK: llilh %r2, 32768
; CHECK-NEXT: br %r14
  ret i64 2147483648
}

; Check the next value up again, which should use LLILF.
define i64 @f9() {
; CHECK-LABEL: f9:
; CHECK: llilf %r2, 2147483649
; CHECK-NEXT: br %r14
  ret i64 2147483649
}

; Check the high end of the LLILH range.
define i64 @f10() {
; CHECK-LABEL: f10:
; CHECK: llilh %r2, 65535
; CHECK-NEXT: br %r14
  ret i64 4294901760
}

; Check the next value up, which must use LLILF.
define i64 @f11() {
; CHECK-LABEL: f11:
; CHECK: llilf %r2, 4294901761
; CHECK-NEXT: br %r14
  ret i64 4294901761
}

; Check the high end of the LLILF range.
define i64 @f12() {
; CHECK-LABEL: f12:
; CHECK: llilf %r2, 4294967295
; CHECK-NEXT: br %r14
  ret i64 4294967295
}

; Check the lowest useful LLIHL value, which is the next one up.
define i64 @f13() {
; CHECK-LABEL: f13:
; CHECK: llihl %r2, 1
; CHECK-NEXT: br %r14
  ret i64 4294967296
}

; Check the next value up, which must use a combination of two instructions.
define i64 @f14() {
; CHECK-LABEL: f14:
; CHECK: llihl %r2, 1
; CHECK-NEXT: oill %r2, 1
; CHECK-NEXT: br %r14
  ret i64 4294967297
}

; Check the high end of the OILL range.
define i64 @f15() {
; CHECK-LABEL: f15:
; CHECK: llihl %r2, 1
; CHECK-NEXT: oill %r2, 65535
; CHECK-NEXT: br %r14
  ret i64 4295032831
}

; Check the next value up, which should use OILH instead.
define i64 @f16() {
; CHECK-LABEL: f16:
; CHECK: llihl %r2, 1
; CHECK-NEXT: oilh %r2, 1
; CHECK-NEXT: br %r14
  ret i64 4295032832
}

; Check the next value up again, which should use OILF.
define i64 @f17() {
; CHECK-LABEL: f17:
; CHECK: llihl %r2, 1
; CHECK-NEXT: oilf %r2, 65537
; CHECK-NEXT: br %r14
  ret i64 4295032833
}

; Check the high end of the OILH range.
define i64 @f18() {
; CHECK-LABEL: f18:
; CHECK: llihl %r2, 1
; CHECK-NEXT: oilh %r2, 65535
; CHECK-NEXT: br %r14
  ret i64 8589869056
}

; Check the high end of the OILF range.
define i64 @f19() {
; CHECK-LABEL: f19:
; CHECK: llihl %r2, 1
; CHECK-NEXT: oilf %r2, 4294967295
; CHECK-NEXT: br %r14
  ret i64 8589934591
}

; Check the high end of the LLIHL range.
define i64 @f20() {
; CHECK-LABEL: f20:
; CHECK: llihl %r2, 65535
; CHECK-NEXT: br %r14
  ret i64 281470681743360
}

; Check the lowest useful LLIHH value, which is 1<<32 greater than the above.
define i64 @f21() {
; CHECK-LABEL: f21:
; CHECK: llihh %r2, 1
; CHECK-NEXT: br %r14
  ret i64 281474976710656
}

; Check the lowest useful LLIHF value, which is 1<<32 greater again.
define i64 @f22() {
; CHECK-LABEL: f22:
; CHECK: llihf %r2, 65537
; CHECK-NEXT: br %r14
  ret i64 281479271677952
}

; Check the highest end of the LLIHH range.
define i64 @f23() {
; CHECK-LABEL: f23:
; CHECK: llihh %r2, 65535
; CHECK-NEXT: br %r14
  ret i64 -281474976710656
}

; Check the next value up, which must use OILL too.
define i64 @f24() {
; CHECK-LABEL: f24:
; CHECK: llihh %r2, 65535
; CHECK-NEXT: oill %r2, 1
; CHECK-NEXT: br %r14
  ret i64 -281474976710655
}

; Check the high end of the LLIHF range.
define i64 @f25() {
; CHECK-LABEL: f25:
; CHECK: llihf %r2, 4294967295
; CHECK-NEXT: br %r14
  ret i64 -4294967296
}

; Check -1.
define i64 @f26() {
; CHECK-LABEL: f26:
; CHECK: lghi %r2, -1
; CHECK-NEXT: br %r14
  ret i64 -1
}

; Check the low end of the LGHI range.
define i64 @f27() {
; CHECK-LABEL: f27:
; CHECK: lghi %r2, -32768
; CHECK-NEXT: br %r14
  ret i64 -32768
}

; Check the next value down, which must use LGFI instead.
define i64 @f28() {
; CHECK-LABEL: f28:
; CHECK: lgfi %r2, -32769
; CHECK-NEXT: br %r14
  ret i64 -32769
}

; Check the low end of the LGFI range.
define i64 @f29() {
; CHECK-LABEL: f29:
; CHECK: lgfi %r2, -2147483648
; CHECK-NEXT: br %r14
  ret i64 -2147483648
}

; Check the next value down, which needs a two-instruction sequence.
define i64 @f30() {
; CHECK-LABEL: f30:
; CHECK: llihf %r2, 4294967295
; CHECK-NEXT: oilf %r2, 2147483647
; CHECK-NEXT: br %r14
  ret i64 -2147483649
}

; Check that constant loads are rematerialized.
define i64 @f31() {
; CHECK-LABEL: f31:
; CHECK-DAG: lghi %r2, 42
; CHECK-DAG: lgfi %r3, 65537
; CHECK-DAG: llilf %r4, 2147483649
; CHECK-DAG: llihf %r5, 65537
; CHECK: brasl %r14, foo@PLT
; CHECK-DAG: llill %r2, 32768
; CHECK-DAG: llilh %r3, 1
; CHECK-DAG: llihl %r4, 1
; CHECK-DAG: llihh %r5, 1
; CHECK: brasl %r14, foo@PLT
; CHECK-DAG: lghi %r2, 42
; CHECK-DAG: lgfi %r3, 65537
; CHECK-DAG: llilf %r4, 2147483649
; CHECK-DAG: llihf %r5, 65537
; CHECK: brasl %r14, foo@PLT
; CHECK-DAG: llill %r2, 32768
; CHECK-DAG: llilh %r3, 1
; CHECK-DAG: llihl %r4, 1
; CHECK-DAG: llihh %r5, 1
; CHECK: brasl %r14, foo@PLT
; CHECK: lghi %r2, 42
; CHECK: br %r14
  call void @foo(i64 42, i64 65537, i64 2147483649, i64 281479271677952)
  call void @foo(i64 32768, i64 65536, i64 4294967296, i64 281474976710656)
  call void @foo(i64 42, i64 65537, i64 2147483649, i64 281479271677952)
  call void @foo(i64 32768, i64 65536, i64 4294967296, i64 281474976710656)
  ret i64 42
}

; Verify that we do not crash on OR with two constant inputs
; (this was PR34859).
define i64 @f32(i64 *%ptr) {
; CHECK-LABEL: f32:
; CHECK: llihf %r1, 918324340
; CHECK: oilf %r1, 1806197964
; CHECK: la %r0, 1(%r1)
  store i64 -1, i64* %ptr, align 8
  %1 = load i64, i64* %ptr, align 8
  %2 = icmp ne i64 %1, 0
  %3 = zext i1 %2 to i64
  %4 = or i64 %3, 3944173009226982604
  store i64 %4, i64* %ptr, align 8
  ret i64 3944173009226982604
}

; Check that huge constants can be loaded during isel pseudo expansion. This
; is the iteration count loaded into a register after dividing by 256.
define void @f33(i8* %Src, i8* %Dst)  {
; CHECK-LABEL: f33:
; CHECK: iihf    %r0, 1
; CHECK: iilf    %r0, 1
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %Src, i8* %Dst, i64 1099511628032, i1 false)
  ret void
}

define void @f34(i8* %Src, i8* %Dst)  {
; CHECK-LABEL: f34:
; CHECK: iihf    %r0, 2
; CHECK: iilf    %r0, 0
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %Src, i8* %Dst, i64 2199023255552, i1 false)
  ret void
}

define void @f35(i8* %Src, i8* %Dst)  {
; CHECK-LABEL: f35:
; CHECK: iihf    %r0, 8388607
; CHECK: iilf    %r0, 4294967295
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %Src, i8* %Dst, i64 9223372036854775800, i1 false)
  ret void
}
