; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 < %s
target triple = "powerpc64-unknown-linux-gnu"

define void @autogen_SD30723(i32) {
BB:
  br label %CF80

CF80:                                             ; preds = %CF80, %BB
  %B = ashr i32 %0, -1
  br i1 undef, label %CF80, label %CF84

CF84:                                             ; preds = %CF84, %CF80
  %Cmp62 = icmp sge i32 undef, %B
  br i1 %Cmp62, label %CF84, label %CF85

CF85:                                             ; preds = %CF85, %CF84
  br label %CF85
}
