; Test that HWASan remove writeonly and memory(*) attributes from instrumented functions.
; RUN: opt -S -passes=hwasan %s | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-android30"

; CHECK: declare dso_local void @test_argmemwriteonly(ptr nocapture noundef) local_unnamed_addr #0
declare dso_local void @test_argmemwriteonly(ptr nocapture noundef writeonly) local_unnamed_addr #0

; CHECK: declare dso_local void @test_argmemwriteonly2(ptr nocapture noundef) local_unnamed_addr #0
declare dso_local void @test_argmemwriteonly2(ptr nocapture noundef) local_unnamed_addr #0

; CHECK: declare dso_local void @test_writeonly(ptr nocapture noundef) local_unnamed_addr #0
declare dso_local void @test_writeonly(ptr nocapture noundef) local_unnamed_addr #1

; CHECK: declare dso_local void @test_writeonly2(ptr nocapture noundef) local_unnamed_addr #0
declare dso_local void @test_writeonly2(ptr nocapture noundef writeonly) local_unnamed_addr #1

; CHECK: declare dso_local void @test_argmemreadonly(ptr nocapture noundef readonly) local_unnamed_addr #0
declare dso_local void @test_argmemreadonly(ptr nocapture noundef readonly) local_unnamed_addr #2

; CHECK: declare dso_local void @test_argmemreadonly2(ptr nocapture noundef) local_unnamed_addr #0
declare dso_local void @test_argmemreadonly2(ptr nocapture noundef) local_unnamed_addr #2

; CHECK: declare dso_local void @test_readonly(ptr nocapture noundef) local_unnamed_addr #1
declare dso_local void @test_readonly(ptr nocapture noundef) local_unnamed_addr #3

; CHECK: declare dso_local void @test_readonly2(ptr nocapture noundef readonly) local_unnamed_addr #1
declare dso_local void @test_readonly2(ptr nocapture noundef readonly) local_unnamed_addr #3

; CHECK: declare dso_local void @test_writeonly_notcalled_from_sanitized(ptr nocapture noundef readonly) local_unnamed_addr #2
declare dso_local void @test_writeonly_notcalled_from_sanitized(ptr nocapture noundef readonly) local_unnamed_addr #1

define dso_local void @test_sanitized(ptr %p) sanitize_hwaddress {
entry:
    call void @test_argmemwriteonly(ptr %p)
    call void @test_argmemwriteonly2(ptr %p)
    call void @test_writeonly(ptr %p)
    call void @test_writeonly2(ptr %p)
    call void @test_argmemreadonly(ptr %p)
    call void @test_argmemreadonly2(ptr %p)
    call void @test_readonly(ptr %p)
    call void @test_readonly2(ptr %p)
    ret void
}

define dso_local void @test_not_sanitized(ptr %p) {
entry:
; CHECK: call void @test_argmemwriteonly(ptr writeonly %p) #5
    call void @test_argmemwriteonly(ptr %p)
; CHECK: call void @test_argmemwriteonly2(ptr %p) #5
    call void @test_argmemwriteonly2(ptr %p)
; CHECK: call void @test_writeonly(ptr %p) #6
    call void @test_writeonly(ptr %p)
; CHECK: call void @test_writeonly2(ptr writeonly %p) #6
    call void @test_writeonly2(ptr %p)
; CHECK: call void @test_argmemreadonly(ptr %p) #7
    call void @test_argmemreadonly(ptr %p)
; CHECK: call void @test_argmemreadonly2(ptr %p) #7
    call void @test_argmemreadonly2(ptr %p)
; CHECK: call void @test_readonly(ptr %p)
    call void @test_readonly(ptr %p)
; CHECK: call void @test_readonly2(ptr %p)
    call void @test_readonly2(ptr %p)
; CHECK: call void @test_writeonly_notcalled_from_sanitized(ptr %p)
    call void @test_writeonly_notcalled_from_sanitized(ptr %p)
    ret void
}

; CHECK: attributes #0 = { nobuiltin uwtable }
; CHECK: attributes #1 = { memory(read) uwtable }
; CHECK: attributes #2 = { memory(write) uwtable }
; CHECK: attributes #3 = { sanitize_hwaddress }
; CHECK: attributes #4 = { nounwind }
; CHECK: attributes #5 = { memory(argmem: write) }
; CHECK: attributes #6 = { memory(write) }
; CHECK: attributes #7 = { memory(argmem: read) }

attributes #0 = { memory(argmem: write) uwtable }
attributes #1 = { memory(write) uwtable }
attributes #2 = { memory(argmem: read) uwtable }
attributes #3 = { memory(read) uwtable }
