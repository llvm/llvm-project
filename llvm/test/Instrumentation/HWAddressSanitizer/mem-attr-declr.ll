; Test that HWASan removes memory attributes from lib functions that are called
; from sanitized functions. We are assuming interceptors or a instrumented
; libc, in which case these attributes are wrong.
;
; RUN: opt -S -passes=hwasan %s | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-android30"

; CHECK: declare dso_local void @strncpy(ptr nocapture noundef) local_unnamed_addr #0
declare dso_local void @strncpy(ptr nocapture noundef writeonly) local_unnamed_addr #0

; CHECK: declare dso_local void @strncmp(ptr nocapture noundef) local_unnamed_addr #0
declare dso_local void @strncmp(ptr nocapture noundef) local_unnamed_addr #0

; CHECK: declare dso_local void @strpbrk(ptr nocapture noundef) local_unnamed_addr #0
declare dso_local void @strpbrk(ptr nocapture noundef) local_unnamed_addr #1

; CHECK: declare dso_local void @strnlen(ptr nocapture noundef) local_unnamed_addr #0
declare dso_local void @strnlen(ptr nocapture noundef writeonly) local_unnamed_addr #1

; CHECK: declare dso_local void @strstr(ptr nocapture noundef readonly) local_unnamed_addr #0
declare dso_local void @strstr(ptr nocapture noundef readonly) local_unnamed_addr #2

; CHECK: declare dso_local void @strspn(ptr nocapture noundef) local_unnamed_addr #0
declare dso_local void @strspn(ptr nocapture noundef) local_unnamed_addr #2

; CHECK: declare dso_local void @strtof(ptr nocapture noundef) local_unnamed_addr #1
declare dso_local void @strtof(ptr nocapture noundef) local_unnamed_addr #3

; CHECK: declare dso_local void @strtod(ptr nocapture noundef readonly) local_unnamed_addr #1
declare dso_local void @strtod(ptr nocapture noundef readonly) local_unnamed_addr #3

; CHECK: declare dso_local void @strndup(ptr nocapture noundef readonly) local_unnamed_addr #2
declare dso_local void @strndup(ptr nocapture noundef readonly) local_unnamed_addr #1

define dso_local void @test_sanitized(ptr %p) sanitize_hwaddress {
entry:
    call void @strncpy(ptr %p)
    call void @strncmp(ptr %p)
    call void @strpbrk(ptr %p)
    call void @strnlen(ptr %p)
    call void @strstr(ptr %p)
    call void @strspn(ptr %p)
    call void @strtof(ptr %p)
    call void @strtod(ptr %p)
    ret void
}

define dso_local void @test_not_sanitized(ptr %p) {
entry:
; CHECK: call void @strncpy(ptr writeonly %p) #5
    call void @strncpy(ptr %p)
; CHECK: call void @strncmp(ptr %p) #5
    call void @strncmp(ptr %p)
; CHECK: call void @strpbrk(ptr %p) #6
    call void @strpbrk(ptr %p)
; CHECK: call void @strnlen(ptr writeonly %p) #6
    call void @strnlen(ptr %p)
; CHECK: call void @strstr(ptr %p) #8
    call void @strstr(ptr %p)
; CHECK: call void @strspn(ptr %p) #8
    call void @strspn(ptr %p)
; CHECK: call void @strtof(ptr %p)
    call void @strtof(ptr %p)
; CHECK: call void @strtod(ptr %p)
    call void @strtod(ptr %p)
; CHECK: call void @strndup(ptr %p)
    call void @strndup(ptr %p)
    ret void
}

; CHECK: attributes #0 = { nobuiltin uwtable }
; CHECK: attributes #1 = { memory(read) uwtable }
; CHECK: attributes #2 = { memory(write) uwtable }
; CHECK: attributes #3 = { sanitize_hwaddress }
; CHECK: attributes #4 = { nounwind }
; CHECK: attributes #5 = { memory(argmem: write) }
; CHECK: attributes #6 = { memory(write) }
; CHECK: attributes #7 = { nobuiltin memory(write) }
; CHECK: attributes #8 = { memory(argmem: read) }

attributes #0 = { memory(argmem: write) uwtable }
attributes #1 = { memory(write) uwtable }
attributes #2 = { memory(argmem: read) uwtable }
attributes #3 = { memory(read) uwtable }
