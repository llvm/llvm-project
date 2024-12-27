; RUN: opt < %s -passes='require<profile-summary>,function(lower-allow-check)' -lower-allow-check-random-rate=1 -pass-remarks=lower-allow-check -pass-remarks-missed=lower-allow-check -S 2>&1 | FileCheck %s
; RUN: opt < %s -passes='require<profile-summary>,function(lower-allow-check)' -lower-allow-check-random-rate=0 -pass-remarks=lower-allow-check -pass-remarks-missed=lower-allow-check -S 2>&1 | FileCheck %s --check-prefixes=REMOVE

; CHECK: remark: <unknown>:0:0: Allowed check: Kind=test_check F=test_runtime BB=entry1
; CHECK: remark: <unknown>:0:0: Allowed check: Kind=7 F=test_ubsan BB=entry2

; REMOVE: remark: <unknown>:0:0: Removed check: Kind=test_check F=test_runtime BB=entry1
; REMOVE: remark: <unknown>:0:0: Removed check: Kind=7 F=test_ubsan BB=entry2

target triple = "x86_64-pc-linux-gnu"

define i1 @test_runtime() local_unnamed_addr {
entry1:
  %allow = call i1 @llvm.allow.runtime.check(metadata !"test_check")
  ret i1 %allow
}

declare i1 @llvm.allow.runtime.check(metadata) nounwind

define i1 @test_ubsan() local_unnamed_addr {
entry2:
  %allow = call i1 @llvm.allow.ubsan.check(i8 7)
  ret i1 %allow
}
