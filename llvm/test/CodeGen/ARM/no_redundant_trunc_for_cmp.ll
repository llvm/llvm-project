; This test check if redundant truncate for eq/ne cmp is skipped during code gen.
;RUN: llc -mtriple=thumbv7-eabi < %s | FileCheck %s

define void @test_zero(i16 signext %x) optsize {
;CHECK-LABEL: test_zero
entry:
  %tobool = icmp eq i16 %x, 0
  br i1 %tobool, label %if.else, label %if.then
;CHECK-NOT: movw {{.*}}, #65535
;CHECK: cbz r0,
if.then:                                          ; preds = %entry
  tail call void bitcast (void (...)* @foo1 to void ()*)()
  br label %if.end

if.else:                                          ; preds = %entry
  tail call void bitcast (void (...)* @foo2 to void ()*)()
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  ret void
}

define void @test_i8_nonzero(i18 signext %x) optsize {
;CHECK-LABEL: test_i8_nonzero
entry:
  %tobool = icmp eq i18 %x, 150
  br i1 %tobool, label %if.else, label %if.then
;CHECK-NOT: bfc
;CHECK: cmp r{{[0-9]+}}, #150
if.then:                                          ; preds = %entry
  tail call void bitcast (void (...)* @foo1 to void ()*)()
  br label %if.end

if.else:                                          ; preds = %entry
  tail call void bitcast (void (...)* @foo2 to void ()*)()
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  ret void
}

define void @test_i8_i16(i8 signext %x) optsize {
;CHECK-LABEL: test_i8_i16
entry:
  %x16 = sext i8 %x to i16
  %tobool = icmp eq i16 %x16, 300
  br i1 %tobool, label %if.else, label %if.then
;CHECK-NOT: uxth r0, r0
;CHECK: cmp.w r0, #300
if.then:                                          ; preds = %entry
  tail call void bitcast (void (...)* @foo1 to void ()*)()
  br label %if.end

if.else:                                          ; preds = %entry
  tail call void bitcast (void (...)* @foo2 to void ()*)()
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  ret void
}

define void @test_i16_i8(i16 signext %x) optsize {
;CHECK-LABEL: test_i16_i8
entry:
;CHECK: uxtb [[REG:r[0-9+]]], r0
;CHECK: cmp [[REG]], #128
  %x8 = trunc i16 %x to i8
  %tobool = icmp eq i8 %x8, 128
  br i1 %tobool, label %if.else, label %if.then
if.then:                                          ; preds = %entry
  tail call void bitcast (void (...)* @foo1 to void ()*)()
  br label %if.end

if.else:                                          ; preds = %entry
  tail call void bitcast (void (...)* @foo2 to void ()*)()
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  ret void
}

define void @test_zext_zero(i16 zeroext %x) optsize {
;CHECK-LABEL: test_zext_zero
entry:
  %tobool = icmp eq i16 %x, 0
  br i1 %tobool, label %if.else, label %if.then
;CHECK-NOT: movw {{.*}}, #65535
;CHECK: cbz r0,
if.then:                                          ; preds = %entry
  tail call void bitcast (void (...)* @foo1 to void ()*)()
  br label %if.end

if.else:                                          ; preds = %entry
  tail call void bitcast (void (...)* @foo2 to void ()*)()
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  ret void
}


declare void @foo1(...)
declare void @foo2(...)


