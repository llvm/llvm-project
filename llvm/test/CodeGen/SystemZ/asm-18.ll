; Test high-word operations, using "h" constraints to force a high
; register and "r" constraints to force a low register.
;
; RUN: llc < %s -verify-machineinstrs -mtriple=s390x-linux-gnu -mcpu=z196 \
; RUN:   -no-integrated-as | FileCheck %s

; Test loads and stores involving mixtures of high and low registers.
define void @f1(ptr %ptr1, ptr %ptr2) {
; CHECK-LABEL: f1:
; CHECK-DAG: lfh [[REG1:%r[0-5]]], 0(%r2)
; CHECK-DAG: l [[REG2:%r[0-5]]], 0(%r3)
; CHECK-DAG: lfh [[REG3:%r[0-5]]], 4096(%r2)
; CHECK-DAG: ly [[REG4:%r[0-5]]], 524284(%r3)
; CHECK: blah [[REG1]], [[REG2]], [[REG3]], [[REG4]]
; CHECK-DAG: stfh [[REG1]], 0(%r2)
; CHECK-DAG: st [[REG2]], 0(%r3)
; CHECK-DAG: stfh [[REG3]], 4096(%r2)
; CHECK-DAG: sty [[REG4]], 524284(%r3)
; CHECK: br %r14
  %ptr3 = getelementptr i32, ptr %ptr1, i64 1024
  %ptr4 = getelementptr i32, ptr %ptr2, i64 131071
  %old1 = load i32, ptr %ptr1
  %old2 = load i32, ptr %ptr2
  %old3 = load i32, ptr %ptr3
  %old4 = load i32, ptr %ptr4
  %res = call { i32, i32, i32, i32 } asm "blah $0, $1, $2, $3",
              "=h,=r,=h,=r,0,1,2,3"(i32 %old1, i32 %old2, i32 %old3, i32 %old4)
  %new1 = extractvalue { i32, i32, i32, i32 } %res, 0
  %new2 = extractvalue { i32, i32, i32, i32 } %res, 1
  %new3 = extractvalue { i32, i32, i32, i32 } %res, 2
  %new4 = extractvalue { i32, i32, i32, i32 } %res, 3
  store i32 %new1, ptr %ptr1
  store i32 %new2, ptr %ptr2
  store i32 %new3, ptr %ptr3
  store i32 %new4, ptr %ptr4
  ret void
}

; Test moves involving mixtures of high and low registers.
define i32 @f2(i32 %old) {
; CHECK-LABEL: f2:
; CHECK-DAG: risbhg [[REG1:%r[0-5]]], %r2, 0, 159, 32
; CHECK-DAG: lr %r3, %r2
; CHECK: stepa [[REG1]], %r2, %r3
; CHECK: risbhg {{%r[0-5]}}, [[REG1]], 0, 159, 0
; CHECK: stepb [[REG2:%r[0-5]]]
; CHECK: risblg %r2, [[REG2]], 0, 159, 32
; CHECK: br %r14
  %tmp = call i32 asm "stepa $1, $2, $3",
                      "=h,0,{r2},{r3}"(i32 %old, i32 %old, i32 %old)
  %new = call i32 asm "stepb $1, $2", "=&h,0,h"(i32 %tmp, i32 %tmp)
  ret i32 %new
}

; Test sign-extending 8-bit loads into mixtures of high and low registers.
define void @f3(ptr %ptr1, ptr %ptr2) {
; CHECK-LABEL: f3:
; CHECK-DAG: lbh [[REG1:%r[0-5]]], 0(%r2)
; CHECK-DAG: lb [[REG2:%r[0-5]]], 0(%r3)
; CHECK-DAG: lbh [[REG3:%r[0-5]]], 4096(%r2)
; CHECK-DAG: lb [[REG4:%r[0-5]]], 524287(%r3)
; CHECK: blah [[REG1]], [[REG2]]
; CHECK: br %r14
  %ptr3 = getelementptr i8, ptr %ptr1, i64 4096
  %ptr4 = getelementptr i8, ptr %ptr2, i64 524287
  %val1 = load i8, ptr %ptr1
  %val2 = load i8, ptr %ptr2
  %val3 = load i8, ptr %ptr3
  %val4 = load i8, ptr %ptr4
  %ext1 = sext i8 %val1 to i32
  %ext2 = sext i8 %val2 to i32
  %ext3 = sext i8 %val3 to i32
  %ext4 = sext i8 %val4 to i32
  call void asm sideeffect "blah $0, $1, $2, $3",
                           "h,r,h,r"(i32 %ext1, i32 %ext2, i32 %ext3, i32 %ext4)
  ret void
}

; Test sign-extending 16-bit loads into mixtures of high and low registers.
define void @f4(ptr %ptr1, ptr %ptr2) {
; CHECK-LABEL: f4:
; CHECK-DAG: lhh [[REG1:%r[0-5]]], 0(%r2)
; CHECK-DAG: lh [[REG2:%r[0-5]]], 0(%r3)
; CHECK-DAG: lhh [[REG3:%r[0-5]]], 4096(%r2)
; CHECK-DAG: lhy [[REG4:%r[0-5]]], 524286(%r3)
; CHECK: blah [[REG1]], [[REG2]]
; CHECK: br %r14
  %ptr3 = getelementptr i16, ptr %ptr1, i64 2048
  %ptr4 = getelementptr i16, ptr %ptr2, i64 262143
  %val1 = load i16, ptr %ptr1
  %val2 = load i16, ptr %ptr2
  %val3 = load i16, ptr %ptr3
  %val4 = load i16, ptr %ptr4
  %ext1 = sext i16 %val1 to i32
  %ext2 = sext i16 %val2 to i32
  %ext3 = sext i16 %val3 to i32
  %ext4 = sext i16 %val4 to i32
  call void asm sideeffect "blah $0, $1, $2, $3",
                           "h,r,h,r"(i32 %ext1, i32 %ext2, i32 %ext3, i32 %ext4)
  ret void
}

; Test zero-extending 8-bit loads into mixtures of high and low registers.
define void @f5(ptr %ptr1, ptr %ptr2) {
; CHECK-LABEL: f5:
; CHECK-DAG: llch [[REG1:%r[0-5]]], 0(%r2)
; CHECK-DAG: llc [[REG2:%r[0-5]]], 0(%r3)
; CHECK-DAG: llch [[REG3:%r[0-5]]], 4096(%r2)
; CHECK-DAG: llc [[REG4:%r[0-5]]], 524287(%r3)
; CHECK: blah [[REG1]], [[REG2]]
; CHECK: br %r14
  %ptr3 = getelementptr i8, ptr %ptr1, i64 4096
  %ptr4 = getelementptr i8, ptr %ptr2, i64 524287
  %val1 = load i8, ptr %ptr1
  %val2 = load i8, ptr %ptr2
  %val3 = load i8, ptr %ptr3
  %val4 = load i8, ptr %ptr4
  %ext1 = zext i8 %val1 to i32
  %ext2 = zext i8 %val2 to i32
  %ext3 = zext i8 %val3 to i32
  %ext4 = zext i8 %val4 to i32
  call void asm sideeffect "blah $0, $1, $2, $3",
                           "h,r,h,r"(i32 %ext1, i32 %ext2, i32 %ext3, i32 %ext4)
  ret void
}

; Test zero-extending 16-bit loads into mixtures of high and low registers.
define void @f6(ptr %ptr1, ptr %ptr2) {
; CHECK-LABEL: f6:
; CHECK-DAG: llhh [[REG1:%r[0-5]]], 0(%r2)
; CHECK-DAG: llh [[REG2:%r[0-5]]], 0(%r3)
; CHECK-DAG: llhh [[REG3:%r[0-5]]], 4096(%r2)
; CHECK-DAG: llh [[REG4:%r[0-5]]], 524286(%r3)
; CHECK: blah [[REG1]], [[REG2]]
; CHECK: br %r14
  %ptr3 = getelementptr i16, ptr %ptr1, i64 2048
  %ptr4 = getelementptr i16, ptr %ptr2, i64 262143
  %val1 = load i16, ptr %ptr1
  %val2 = load i16, ptr %ptr2
  %val3 = load i16, ptr %ptr3
  %val4 = load i16, ptr %ptr4
  %ext1 = zext i16 %val1 to i32
  %ext2 = zext i16 %val2 to i32
  %ext3 = zext i16 %val3 to i32
  %ext4 = zext i16 %val4 to i32
  call void asm sideeffect "blah $0, $1, $2, $3",
                           "h,r,h,r"(i32 %ext1, i32 %ext2, i32 %ext3, i32 %ext4)
  ret void
}

; Test truncating stores of high and low registers into 8-bit memory.
define void @f7(ptr %ptr1, ptr %ptr2) {
; CHECK-LABEL: f7:
; CHECK: blah [[REG1:%r[0-5]]], [[REG2:%r[0-5]]]
; CHECK-DAG: stch [[REG1]], 0(%r2)
; CHECK-DAG: stc [[REG2]], 0(%r3)
; CHECK-DAG: stch [[REG1]], 4096(%r2)
; CHECK-DAG: stcy [[REG2]], 524287(%r3)
; CHECK: br %r14
  %res = call { i32, i32 } asm "blah $0, $1", "=h,=r"()
  %res1 = extractvalue { i32, i32 } %res, 0
  %res2 = extractvalue { i32, i32 } %res, 1
  %trunc1 = trunc i32 %res1 to i8
  %trunc2 = trunc i32 %res2 to i8
  %ptr3 = getelementptr i8, ptr %ptr1, i64 4096
  %ptr4 = getelementptr i8, ptr %ptr2, i64 524287
  store i8 %trunc1, ptr %ptr1
  store i8 %trunc2, ptr %ptr2
  store i8 %trunc1, ptr %ptr3
  store i8 %trunc2, ptr %ptr4
  ret void
}

; Test truncating stores of high and low registers into 16-bit memory.
define void @f8(ptr %ptr1, ptr %ptr2) {
; CHECK-LABEL: f8:
; CHECK: blah [[REG1:%r[0-5]]], [[REG2:%r[0-5]]]
; CHECK-DAG: sthh [[REG1]], 0(%r2)
; CHECK-DAG: sth [[REG2]], 0(%r3)
; CHECK-DAG: sthh [[REG1]], 4096(%r2)
; CHECK-DAG: sthy [[REG2]], 524286(%r3)
; CHECK: br %r14
  %res = call { i32, i32 } asm "blah $0, $1", "=h,=r"()
  %res1 = extractvalue { i32, i32 } %res, 0
  %res2 = extractvalue { i32, i32 } %res, 1
  %trunc1 = trunc i32 %res1 to i16
  %trunc2 = trunc i32 %res2 to i16
  %ptr3 = getelementptr i16, ptr %ptr1, i64 2048
  %ptr4 = getelementptr i16, ptr %ptr2, i64 262143
  store i16 %trunc1, ptr %ptr1
  store i16 %trunc2, ptr %ptr2
  store i16 %trunc1, ptr %ptr3
  store i16 %trunc2, ptr %ptr4
  ret void
}

; Test zero extensions from 8 bits between mixtures of high and low registers.
define i32 @f9(i8 %val1, i8 %val2) {
; CHECK-LABEL: f9:
; CHECK-DAG: risbhg [[REG1:%r[0-5]]], %r2, 24, 159, 32
; CHECK-DAG: llcr [[REG2:%r[0-5]]], %r3
; CHECK: stepa [[REG1]], [[REG2]]
; CHECK: risbhg [[REG3:%r[0-5]]], [[REG1]], 24, 159, 0
; CHECK: stepb [[REG3]]
; CHECK: risblg %r2, [[REG3]], 24, 159, 32
; CHECK: br %r14
  %ext1 = zext i8 %val1 to i32
  %ext2 = zext i8 %val2 to i32
  %val3 = call i8 asm sideeffect "stepa $0, $1", "=h,0,r"(i32 %ext1, i32 %ext2)
  %ext3 = zext i8 %val3 to i32
  %val4 = call i8 asm sideeffect "stepb $0", "=h,0"(i32 %ext3)
  %ext4 = zext i8 %val4 to i32
  ret i32 %ext4
}

; Test zero extensions from 16 bits between mixtures of high and low registers.
define i32 @f10(i16 %val1, i16 %val2) {
; CHECK-LABEL: f10:
; CHECK-DAG: risbhg [[REG1:%r[0-5]]], %r2, 16, 159, 32
; CHECK-DAG: llhr [[REG2:%r[0-5]]], %r3
; CHECK: stepa [[REG1]], [[REG2]]
; CHECK: risbhg [[REG3:%r[0-5]]], [[REG1]], 16, 159, 0
; CHECK: stepb [[REG3]]
; CHECK: risblg %r2, [[REG3]], 16, 159, 32
; CHECK: br %r14
  %ext1 = zext i16 %val1 to i32
  %ext2 = zext i16 %val2 to i32
  %val3 = call i16 asm sideeffect "stepa $0, $1", "=h,0,r"(i32 %ext1, i32 %ext2)
  %ext3 = zext i16 %val3 to i32
  %val4 = call i16 asm sideeffect "stepb $0", "=h,0"(i32 %ext3)
  %ext4 = zext i16 %val4 to i32
  ret i32 %ext4
}

; Test loads of 16-bit constants into mixtures of high and low registers.
define void @f11() {
; CHECK-LABEL: f11:
; CHECK-DAG: iihf [[REG1:%r[0-5]]], 4294934529
; CHECK-DAG: lhi [[REG2:%r[0-5]]], -32768
; CHECK-DAG: llihl [[REG3:%r[0-5]]], 32766
; CHECK-DAG: lhi [[REG4:%r[0-5]]], 32767
; CHECK: blah [[REG1]], [[REG2]], [[REG3]], [[REG4]]
; CHECK: br %r14
  call void asm sideeffect "blah $0, $1, $2, $3",
                           "h,r,h,r"(i32 -32767, i32 -32768,
                                     i32 32766, i32 32767)
  ret void
}

; Test loads of unsigned constants into mixtures of high and low registers.
; For stepc, we expect the h and r operands to be paired by the register
; allocator.  It doesn't really matter which comes first: LLILL/IIHF would
; be just as good.
define void @f12() {
; CHECK-LABEL: f12:
; CHECK-DAG: llihl [[REG1:%r[0-5]]], 32768
; CHECK-DAG: llihl [[REG2:%r[0-5]]], 65535
; CHECK-DAG: llihh [[REG3:%r[0-5]]], 1
; CHECK-DAG: llihh [[REG4:%r[0-5]]], 65535
; CHECK: stepa [[REG1]], [[REG2]], [[REG3]], [[REG4]]
; CHECK-DAG: llill [[REG1:%r[0-5]]], 32769
; CHECK-DAG: llill [[REG2:%r[0-5]]], 65534
; CHECK-DAG: llilh [[REG3:%r[0-5]]], 2
; CHECK-DAG: llilh [[REG4:%r[0-5]]], 65534
; CHECK: stepb [[REG1]], [[REG2]], [[REG3]], [[REG4]]
; CHECK-DAG: llihl [[REG1:%r[0-5]]], 32770
; CHECK-DAG: iilf [[REG1]], 65533
; CHECK-DAG: llihh [[REG2:%r[0-5]]], 4
; CHECK-DAG: iilf [[REG2]], 524288
; CHECK: stepc [[REG1]], [[REG1]], [[REG2]], [[REG2]]
; CHECK-DAG: iihf [[REG1:%r[0-5]]], 3294967296
; CHECK-DAG: iilf [[REG2:%r[0-5]]], 4294567296
; CHECK-DAG: iihf [[REG3:%r[0-5]]], 1000000000
; CHECK-DAG: iilf [[REG4:%r[0-5]]], 400000
; CHECK: stepd [[REG1]], [[REG2]], [[REG3]], [[REG4]]
; CHECK: br %r14
  call void asm sideeffect "stepa $0, $1, $2, $3",
                           "h,h,h,h"(i32 32768, i32 65535,
                                     i32 65536, i32 -65536)
  call void asm sideeffect "stepb $0, $1, $2, $3",
                           "r,r,r,r"(i32 32769, i32 65534,
                                     i32 131072, i32 -131072)
  call void asm sideeffect "stepc $0, $1, $2, $3",
                           "h,r,h,r"(i32 32770, i32 65533,
                                     i32 262144, i32 524288)
  call void asm sideeffect "stepd $0, $1, $2, $3",
                           "h,r,h,r"(i32 -1000000000, i32 -400000,
                                     i32 1000000000, i32 400000)
  ret void
}

; Test selects involving high registers.
; Note that we prefer to use a LOCR and move the result to a high register.
define void @f13(i32 %x, i32 %y) {
; CHECK-LABEL: f13:
; CHECK-DAG: chi %r2, 0
; CHECK-DAG: iilf [[REG1:%r[0-5]]], 2102030405
; CHECK-DAG: lhi [[REG2:%r[0-5]]], 0
; CHECK: locre [[REG1]], [[REG2]]
; CHECK: risbhg [[REG:%r[0-5]]], [[REG1]], 0, 159, 32
; CHECK: blah [[REG]]
; CHECK: br %r14
  %cmp = icmp eq i32 %x, 0
  %val = select i1 %cmp, i32 0, i32 2102030405
  call void asm sideeffect "blah $0", "h"(i32 %val)
  ret void
}

; Test selects involving low registers.
define void @f14(i32 %x, i32 %y) {
; CHECK-LABEL: f14:
; CHECK-DAG: chi %r2, 0
; CHECK-DAG: iilf [[REG:%r[0-5]]], 2102030405
; CHECK-DAG: lhi [[REG1:%r[0-5]]], 0
; CHECK: locre [[REG]], [[REG1]]
; CHECK: blah [[REG]]
; CHECK: br %r14
  %cmp = icmp eq i32 %x, 0
  %val = select i1 %cmp, i32 0, i32 2102030405
  call void asm sideeffect "blah $0", "r"(i32 %val)
  ret void
}

; Test immediate insertion involving high registers.
define void @f15() {
; CHECK-LABEL: f15:
; CHECK: stepa [[REG:%r[0-5]]]
; CHECK: iihh [[REG]], 4660
; CHECK: stepb [[REG]]
; CHECK: iihl [[REG]], 34661
; CHECK: stepc [[REG]]
; CHECK: br %r14
  %res1 = call i32 asm "stepa $0", "=h"()
  %and1 = and i32 %res1, 65535
  %or1 = or i32 %and1, 305397760
  %res2 = call i32 asm "stepb $0, $1", "=h,h"(i32 %or1)
  %and2 = and i32 %res2, -65536
  %or2 = or i32 %and2, 34661
  call void asm sideeffect "stepc $0", "h"(i32 %or2)
  ret void
}

; Test immediate insertion involving low registers.
define void @f16() {
; CHECK-LABEL: f16:
; CHECK: stepa [[REG:%r[0-5]]]
; CHECK: iilh [[REG]], 4660
; CHECK: stepb [[REG]]
; CHECK: iill [[REG]], 34661
; CHECK: stepc [[REG]]
; CHECK: br %r14
  %res1 = call i32 asm "stepa $0", "=r"()
  %and1 = and i32 %res1, 65535
  %or1 = or i32 %and1, 305397760
  %res2 = call i32 asm "stepb $0, $1", "=r,r"(i32 %or1)
  %and2 = and i32 %res2, -65536
  %or2 = or i32 %and2, 34661
  call void asm sideeffect "stepc $0", "r"(i32 %or2)
  ret void
}

; Test immediate OR involving high registers.
define void @f17() {
; CHECK-LABEL: f17:
; CHECK: stepa [[REG:%r[0-5]]]
; CHECK: oihh [[REG]], 4660
; CHECK: stepb [[REG]]
; CHECK: oihl [[REG]], 34661
; CHECK: stepc [[REG]]
; CHECK: oihf [[REG]], 12345678
; CHECK: stepd [[REG]]
; CHECK: br %r14
  %res1 = call i32 asm "stepa $0", "=h"()
  %or1 = or i32 %res1, 305397760
  %res2 = call i32 asm "stepb $0, $1", "=h,h"(i32 %or1)
  %or2 = or i32 %res2, 34661
  %res3 = call i32 asm "stepc $0, $1", "=h,h"(i32 %or2)
  %or3 = or i32 %res3, 12345678
  call void asm sideeffect "stepd $0", "h"(i32 %or3)
  ret void
}

; Test immediate OR involving low registers.
define void @f18() {
; CHECK-LABEL: f18:
; CHECK: stepa [[REG:%r[0-5]]]
; CHECK: oilh [[REG]], 4660
; CHECK: stepb [[REG]]
; CHECK: oill [[REG]], 34661
; CHECK: stepc [[REG]]
; CHECK: oilf [[REG]], 12345678
; CHECK: stepd [[REG]]
; CHECK: br %r14
  %res1 = call i32 asm "stepa $0", "=r"()
  %or1 = or i32 %res1, 305397760
  %res2 = call i32 asm "stepb $0, $1", "=r,r"(i32 %or1)
  %or2 = or i32 %res2, 34661
  %res3 = call i32 asm "stepc $0, $1", "=r,r"(i32 %or2)
  %or3 = or i32 %res3, 12345678
  call void asm sideeffect "stepd $0", "r"(i32 %or3)
  ret void
}

; Test immediate XOR involving high registers.
define void @f19() {
; CHECK-LABEL: f19:
; CHECK: stepa [[REG:%r[0-5]]]
; CHECK: xihf [[REG]], 305397760
; CHECK: stepb [[REG]]
; CHECK: xihf [[REG]], 34661
; CHECK: stepc [[REG]]
; CHECK: xihf [[REG]], 12345678
; CHECK: stepd [[REG]]
; CHECK: br %r14
  %res1 = call i32 asm "stepa $0", "=h"()
  %xor1 = xor i32 %res1, 305397760
  %res2 = call i32 asm "stepb $0, $1", "=h,h"(i32 %xor1)
  %xor2 = xor i32 %res2, 34661
  %res3 = call i32 asm "stepc $0, $1", "=h,h"(i32 %xor2)
  %xor3 = xor i32 %res3, 12345678
  call void asm sideeffect "stepd $0", "h"(i32 %xor3)
  ret void
}

; Test immediate XOR involving low registers.
define void @f20() {
; CHECK-LABEL: f20:
; CHECK: stepa [[REG:%r[0-5]]]
; CHECK: xilf [[REG]], 305397760
; CHECK: stepb [[REG]]
; CHECK: xilf [[REG]], 34661
; CHECK: stepc [[REG]]
; CHECK: xilf [[REG]], 12345678
; CHECK: stepd [[REG]]
; CHECK: br %r14
  %res1 = call i32 asm "stepa $0", "=r"()
  %xor1 = xor i32 %res1, 305397760
  %res2 = call i32 asm "stepb $0, $1", "=r,r"(i32 %xor1)
  %xor2 = xor i32 %res2, 34661
  %res3 = call i32 asm "stepc $0, $1", "=r,r"(i32 %xor2)
  %xor3 = xor i32 %res3, 12345678
  call void asm sideeffect "stepd $0", "r"(i32 %xor3)
  ret void
}

; Test two-operand immediate AND involving high registers.
define void @f21() {
; CHECK-LABEL: f21:
; CHECK: stepa [[REG:%r[0-5]]]
; CHECK: nihh [[REG]], 4096
; CHECK: stepb [[REG]]
; CHECK: nihl [[REG]], 57536
; CHECK: stepc [[REG]]
; CHECK: nihf [[REG]], 12345678
; CHECK: stepd [[REG]]
; CHECK: br %r14
  %res1 = call i32 asm "stepa $0", "=h"()
  %and1 = and i32 %res1, 268500991
  %res2 = call i32 asm "stepb $0, $1", "=h,h"(i32 %and1)
  %and2 = and i32 %res2, -8000
  %res3 = call i32 asm "stepc $0, $1", "=h,h"(i32 %and2)
  %and3 = and i32 %res3, 12345678
  call void asm sideeffect "stepd $0", "h"(i32 %and3)
  ret void
}

; Test two-operand immediate AND involving low registers.
define void @f22() {
; CHECK-LABEL: f22:
; CHECK: stepa [[REG:%r[0-5]]]
; CHECK: nilh [[REG]], 4096
; CHECK: stepb [[REG]]
; CHECK: nill [[REG]], 57536
; CHECK: stepc [[REG]]
; CHECK: nilf [[REG]], 12345678
; CHECK: stepd [[REG]]
; CHECK: br %r14
  %res1 = call i32 asm "stepa $0", "=r"()
  %and1 = and i32 %res1, 268500991
  %res2 = call i32 asm "stepb $0, $1", "=r,r"(i32 %and1)
  %and2 = and i32 %res2, -8000
  %res3 = call i32 asm "stepc $0, $1", "=r,r"(i32 %and2)
  %and3 = and i32 %res3, 12345678
  call void asm sideeffect "stepd $0", "r"(i32 %and3)
  ret void
}

; Test three-operand immediate AND involving mixtures of low and high registers.
define i32 @f23(i32 %old) {
; CHECK-LABEL: f23:
; CHECK-DAG: risblg [[REG1:%r[0-5]]], %r2, 28, 158, 0
; CHECK-DAG: risbhg [[REG2:%r[0-5]]], %r2, 24, 158, 32
; CHECK: stepa %r2, [[REG1]], [[REG2]]
; CHECK-DAG: risbhg [[REG3:%r[0-5]]], [[REG2]], 25, 159, 0
; CHECK-DAG: risblg %r2, [[REG2]], 24, 152, 32
; CHECK: stepb [[REG2]], [[REG3]], %r2
; CHECK: br %r14
  %and1 = and i32 %old, 14
  %and2 = and i32 %old, 254
  %res1 = call i32 asm "stepa $1, $2, $3",
                       "=h,r,r,0"(i32 %old, i32 %and1, i32 %and2)
  %and3 = and i32 %res1, 127
  %and4 = and i32 %res1, 128
  %res2 = call i32 asm "stepb $1, $2, $3",
                       "=r,h,h,0"(i32 %res1, i32 %and3, i32 %and4)
  ret i32 %res2
}

; Test RISB[LH]G insertions involving mixtures of high and low registers.
define i32 @f24(i32 %old) {
; CHECK-LABEL: f24:
; CHECK-DAG: risblg [[REG1:%r[0-5]]], %r2, 28, 158, 1
; CHECK-DAG: risbhg [[REG2:%r[0-5]]], %r2, 24, 158, 29
; CHECK: stepa %r2, [[REG1]], [[REG2]]
; CHECK-DAG: risbhg [[REG3:%r[0-5]]], [[REG2]], 25, 159, 62
; CHECK-DAG: risblg %r2, [[REG2]], 24, 152, 37
; CHECK: stepb [[REG2]], [[REG3]], %r2
; CHECK: br %r14
  %shift1 = shl i32 %old, 1
  %and1 = and i32 %shift1, 14
  %shift2 = lshr i32 %old, 3
  %and2 = and i32 %shift2, 254
  %res1 = call i32 asm "stepa $1, $2, $3",
                       "=h,r,r,0"(i32 %old, i32 %and1, i32 %and2)
  %shift3 = lshr i32 %res1, 2
  %and3 = and i32 %shift3, 127
  %shift4 = shl i32 %res1, 5
  %and4 = and i32 %shift4, 128
  %res2 = call i32 asm "stepb $1, $2, $3",
                       "=r,h,h,0"(i32 %res1, i32 %and3, i32 %and4)
  ret i32 %res2
}

; Test TMxx involving mixtures of high and low registers.
define i32 @f25(i32 %old) {
; CHECK-LABEL: f25:
; CHECK-DAG: tmll %r2, 1
; CHECK-DAG: tmlh %r2, 1
; CHECK: stepa [[REG1:%r[0-5]]],
; CHECK-DAG: tmhl [[REG1]], 1
; CHECK-DAG: tmhh [[REG1]], 1
; CHECK: stepb %r2,
; CHECK: br %r14
  %and1 = and i32 %old, 1
  %and2 = and i32 %old, 65536
  %cmp1 = icmp eq i32 %and1, 0
  %cmp2 = icmp eq i32 %and2, 0
  %sel1 = select i1 %cmp1, i32 100, i32 200
  %sel2 = select i1 %cmp2, i32 100, i32 200
  %res1 = call i32 asm "stepa $0, $1, $2",
                       "=h,r,r"(i32 %sel1, i32 %sel2)
  %and3 = and i32 %res1, 1
  %and4 = and i32 %res1, 65536
  %cmp3 = icmp eq i32 %and3, 0
  %cmp4 = icmp eq i32 %and4, 0
  %sel3 = select i1 %cmp3, i32 100, i32 200
  %sel4 = select i1 %cmp4, i32 100, i32 200
  %res2 = call i32 asm "stepb $0, $1, $2",
                       "=r,h,h"(i32 %sel3, i32 %sel4)
  ret i32 %res2
}

; Test two-operand halfword immediate addition involving high registers.
define void @f26() {
; CHECK-LABEL: f26:
; CHECK: stepa [[REG:%r[0-5]]]
; CHECK: aih [[REG]], -32768
; CHECK: stepb [[REG]]
; CHECK: aih [[REG]], 1
; CHECK: stepc [[REG]]
; CHECK: aih [[REG]], 32767
; CHECK: stepd [[REG]]
; CHECK: br %r14
  %res1 = call i32 asm "stepa $0", "=h"()
  %add1 = add i32 %res1, -32768
  %res2 = call i32 asm "stepb $0, $1", "=h,h"(i32 %add1)
  %add2 = add i32 %res2, 1
  %res3 = call i32 asm "stepc $0, $1", "=h,h"(i32 %add2)
  %add3 = add i32 %res3, 32767
  call void asm sideeffect "stepd $0", "h"(i32 %add3)
  ret void
}

; Test two-operand halfword immediate addition involving low registers.
define void @f27() {
; CHECK-LABEL: f27:
; CHECK: stepa [[REG:%r[0-5]]]
; CHECK: ahi [[REG]], -32768
; CHECK: stepb [[REG]]
; CHECK: ahi [[REG]], 1
; CHECK: stepc [[REG]]
; CHECK: ahi [[REG]], 32767
; CHECK: stepd [[REG]]
; CHECK: br %r14
  %res1 = call i32 asm "stepa $0", "=r"()
  %add1 = add i32 %res1, -32768
  %res2 = call i32 asm "stepb $0, $1", "=r,r"(i32 %add1)
  %add2 = add i32 %res2, 1
  %res3 = call i32 asm "stepc $0, $1", "=r,r"(i32 %add2)
  %add3 = add i32 %res3, 32767
  call void asm sideeffect "stepd $0", "r"(i32 %add3)
  ret void
}

; Test three-operand halfword immediate addition involving mixtures of low
; and high registers.  AHIK/RISBHG would be OK too, instead of RISBHG/AIH.
define i32 @f28(i32 %old) {
; CHECK-LABEL: f28:
; CHECK: ahik [[REG1:%r[0-5]]], %r2, 14
; CHECK: stepa %r2, [[REG1]]
; CHECK: risbhg  [[REG1]], [[REG1]], 0, 159, 32
; CHECK: aih     [[REG1]], 254
; CHECK: stepb [[REG1]], [[REG2]]
; CHECK: risbhg [[REG3:%r[0-5]]], [[REG2]], 0, 159, 0
; CHECK: aih [[REG3]], 127
; CHECK: stepc [[REG2]], [[REG3]]
; CHECK: risblg %r2, [[REG3]], 0, 159, 32
; CHECK: ahi %r2, 128
; CHECK: stepd [[REG3]], %r2
; CHECK: br %r14
  %add1 = add i32 %old, 14
  %res1 = call i32 asm "stepa $1, $2",
                       "=r,r,0"(i32 %old, i32 %add1)
  %add2 = add i32 %res1, 254
  %res2 = call i32 asm "stepb $1, $2",
                       "=h,r,0"(i32 %res1, i32 %add2)
  %add3 = add i32 %res2, 127
  %res3 = call i32 asm "stepc $1, $2",
                       "=h,h,0"(i32 %res2, i32 %add3)
  %add4 = add i32 %res3, 128
  %res4 = call i32 asm "stepd $1, $2",
                       "=r,h,0"(i32 %res3, i32 %add4)
  ret i32 %res4
}

; Test large immediate addition involving high registers.
define void @f29() {
; CHECK-LABEL: f29:
; CHECK: stepa [[REG:%r[0-5]]]
; CHECK: aih [[REG]], -32769
; CHECK: stepb [[REG]]
; CHECK: aih [[REG]], 32768
; CHECK: stepc [[REG]]
; CHECK: aih [[REG]], 1000000000
; CHECK: stepd [[REG]]
; CHECK: br %r14
  %res1 = call i32 asm "stepa $0", "=h"()
  %add1 = add i32 %res1, -32769
  %res2 = call i32 asm "stepb $0, $1", "=h,h"(i32 %add1)
  %add2 = add i32 %res2, 32768
  %res3 = call i32 asm "stepc $0, $1", "=h,h"(i32 %add2)
  %add3 = add i32 %res3, 1000000000
  call void asm sideeffect "stepd $0", "h"(i32 %add3)
  ret void
}

; Test large immediate addition involving low registers.
define void @f30() {
; CHECK-LABEL: f30:
; CHECK: stepa [[REG:%r[0-5]]]
; CHECK: afi [[REG]], -32769
; CHECK: stepb [[REG]]
; CHECK: afi [[REG]], 32768
; CHECK: stepc [[REG]]
; CHECK: afi [[REG]], 1000000000
; CHECK: stepd [[REG]]
; CHECK: br %r14
  %res1 = call i32 asm "stepa $0", "=r"()
  %add1 = add i32 %res1, -32769
  %res2 = call i32 asm "stepb $0, $1", "=r,r"(i32 %add1)
  %add2 = add i32 %res2, 32768
  %res3 = call i32 asm "stepc $0, $1", "=r,r"(i32 %add2)
  %add3 = add i32 %res3, 1000000000
  call void asm sideeffect "stepd $0", "r"(i32 %add3)
  ret void
}

; Test large immediate comparison involving high registers.
define i32 @f31() {
; CHECK-LABEL: f31:
; CHECK: stepa [[REG1:%r[0-5]]]
; CHECK: cih [[REG1]], 1000000000
; CHECK: stepb [[REG2:%r[0-5]]]
; CHECK: clih [[REG2]], 1000000000
; CHECK: br %r14
  %res1 = call i32 asm "stepa $0", "=h"()
  %cmp1 = icmp sle i32 %res1, 1000000000
  %sel1 = select i1 %cmp1, i32 0, i32 1
  %res2 = call i32 asm "stepb $0, $1", "=h,r"(i32 %sel1)
  %cmp2 = icmp ule i32 %res2, 1000000000
  %sel2 = select i1 %cmp2, i32 0, i32 1
  ret i32 %sel2
}

; Test large immediate comparison involving low registers.
define i32 @f32() {
; CHECK-LABEL: f32:
; CHECK: stepa [[REG1:%r[0-5]]]
; CHECK: cfi [[REG1]], 1000000000
; CHECK: stepb [[REG2:%r[0-5]]]
; CHECK: clfi [[REG2]], 1000000000
; CHECK: br %r14
  %res1 = call i32 asm "stepa $0", "=r"()
  %cmp1 = icmp sle i32 %res1, 1000000000
  %sel1 = select i1 %cmp1, i32 0, i32 1
  %res2 = call i32 asm "stepb $0, $1", "=r,r"(i32 %sel1)
  %cmp2 = icmp ule i32 %res2, 1000000000
  %sel2 = select i1 %cmp2, i32 0, i32 1
  ret i32 %sel2
}

; Test memory comparison involving high registers.
define void @f33(ptr %ptr1, ptr %ptr2) {
; CHECK-LABEL: f33:
; CHECK: stepa [[REG1:%r[0-5]]]
; CHECK: chf [[REG1]], 0(%r2)
; CHECK: stepb [[REG2:%r[0-5]]]
; CHECK: clhf [[REG2]], 0(%r3)
; CHECK: br %r14
  %res1 = call i32 asm "stepa $0", "=h"()
  %load1 = load i32, ptr %ptr1
  %cmp1 = icmp sle i32 %res1, %load1
  %sel1 = select i1 %cmp1, i32 0, i32 1
  %res2 = call i32 asm "stepb $0, $1", "=h,r"(i32 %sel1)
  %load2 = load i32, ptr %ptr2
  %cmp2 = icmp ule i32 %res2, %load2
  %sel2 = select i1 %cmp2, i32 0, i32 1
  store i32 %sel2, ptr %ptr1
  ret void
}

; Test memory comparison involving low registers.
define void @f34(ptr %ptr1, ptr %ptr2) {
; CHECK-LABEL: f34:
; CHECK: stepa [[REG1:%r[0-5]]]
; CHECK: c [[REG1]], 0(%r2)
; CHECK: stepb [[REG2:%r[0-5]]]
; CHECK: cl [[REG2]], 0(%r3)
; CHECK: br %r14
  %res1 = call i32 asm "stepa $0", "=r"()
  %load1 = load i32, ptr %ptr1
  %cmp1 = icmp sle i32 %res1, %load1
  %sel1 = select i1 %cmp1, i32 0, i32 1
  %res2 = call i32 asm "stepb $0, $1", "=r,r"(i32 %sel1)
  %load2 = load i32, ptr %ptr2
  %cmp2 = icmp ule i32 %res2, %load2
  %sel2 = select i1 %cmp2, i32 0, i32 1
  store i32 %sel2, ptr %ptr1
  ret void
}

; Test immediate addition with overflow involving high registers.
define void @f35() {
; CHECK-LABEL: f35:
; CHECK: stepa [[REG:%r[0-5]]]
; CHECK: aih [[REG]], -32768
; CHECK: ipm [[REGCC:%r[0-5]]]
; CHECK: afi [[REGCC]], 1342177280
; CHECK: srl [[REGCC]], 31
; CHECK: stepb [[REG]], [[REGCC]]
; CHECK: aih [[REG]], 1
; CHECK: ipm [[REGCC:%r[0-5]]]
; CHECK: afi [[REGCC]], 1342177280
; CHECK: srl [[REGCC]], 31
; CHECK: stepc [[REG]], [[REGCC]]
; CHECK: aih [[REG]], 32767
; CHECK: ipm [[REGCC:%r[0-5]]]
; CHECK: afi [[REGCC]], 1342177280
; CHECK: srl [[REGCC]], 31
; CHECK: stepd [[REG]], [[REGCC]]
; CHECK: br %r14
  %res1 = call i32 asm "stepa $0", "=h"()
  %t1 = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %res1, i32 -32768)
  %val1 = extractvalue {i32, i1} %t1, 0
  %obit1 = extractvalue {i32, i1} %t1, 1
  %res2 = call i32 asm "stepb $0, $2", "=h,h,d"(i32 %val1, i1 %obit1)
  %t2 = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %res2, i32 1)
  %val2 = extractvalue {i32, i1} %t2, 0
  %obit2 = extractvalue {i32, i1} %t2, 1
  %res3 = call i32 asm "stepc $0, $2", "=h,h,d"(i32 %val2, i1 %obit2)
  %t3 = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %res3, i32 32767)
  %val3 = extractvalue {i32, i1} %t3, 0
  %obit3 = extractvalue {i32, i1} %t3, 1
  call void asm sideeffect "stepd $0, $1", "h,d"(i32 %val3, i1 %obit3)
  ret void
}

; Test large immediate addition with overflow involving high registers.
define void @f36() {
; CHECK-LABEL: f36:
; CHECK: stepa [[REG:%r[0-5]]]
; CHECK: aih [[REG]], -2147483648
; CHECK: ipm [[REGCC:%r[0-5]]]
; CHECK: afi [[REGCC]], 1342177280
; CHECK: srl [[REGCC]], 31
; CHECK: stepb [[REG]], [[REGCC]]
; CHECK: aih [[REG]], 1
; CHECK: ipm [[REGCC:%r[0-5]]]
; CHECK: afi [[REGCC]], 1342177280
; CHECK: srl [[REGCC]], 31
; CHECK: stepc [[REG]], [[REGCC]]
; CHECK: aih [[REG]], 2147483647
; CHECK: ipm [[REGCC:%r[0-5]]]
; CHECK: afi [[REGCC]], 1342177280
; CHECK: srl [[REGCC]], 31
; CHECK: stepd [[REG]], [[REGCC]]
; CHECK: br %r14
  %res1 = call i32 asm "stepa $0", "=h"()
  %t1 = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %res1, i32 -2147483648)
  %val1 = extractvalue {i32, i1} %t1, 0
  %obit1 = extractvalue {i32, i1} %t1, 1
  %res2 = call i32 asm "stepb $0, $2", "=h,h,d"(i32 %val1, i1 %obit1)
  %t2 = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %res2, i32 1)
  %val2 = extractvalue {i32, i1} %t2, 0
  %obit2 = extractvalue {i32, i1} %t2, 1
  %res3 = call i32 asm "stepc $0, $2", "=h,h,d"(i32 %val2, i1 %obit2)
  %t3 = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %res3, i32 2147483647)
  %val3 = extractvalue {i32, i1} %t3, 0
  %obit3 = extractvalue {i32, i1} %t3, 1
  call void asm sideeffect "stepd $0, $1", "h,d"(i32 %val3, i1 %obit3)
  ret void
}

declare {i32, i1} @llvm.sadd.with.overflow.i32(i32, i32) nounwind readnone

