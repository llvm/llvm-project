; RUN: opt -mtriple=x86_64-unknown-linux-gnu -passes=inferattrs,verify -S %s | FileCheck %s

; A const variadic declaration can already have readnone on its fixed pointer
; arguments.  Libcall inference must not add conflicting access attributes.
; CHECK: declare noundef i32 @snprintf(ptr {{.*}}readnone{{.*}}, i64 noundef, ptr {{.*}}readnone{{.*}}, ...)
declare i32 @snprintf(ptr readnone, i64, ptr readnone, ...)

; The same applies to sprintf, which exercises both inferred access directions.
; CHECK: declare noundef i32 @sprintf(ptr {{.*}}readnone{{.*}}, ptr {{.*}}readnone{{.*}}, ...)
declare i32 @sprintf(ptr readnone, ptr readnone, ...)
