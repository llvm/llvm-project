; RUN: llc -O0 --global-isel=1 %s -o - -verify-machineinstrs
target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

define i1 @foo(i64) {
    %a = sext i64 %0 to i128
    %b = icmp sle i128 %a, 0
    ret i1 %b
}
