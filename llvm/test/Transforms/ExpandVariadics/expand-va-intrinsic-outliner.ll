; RUN: opt -mtriple=i386-unknown-linux-gnu -S --passes=expand-variadics --expand-variadics-override=optimize < %s | FileCheck %s -check-prefix=X86
; RUN: opt -mtriple=x86_64-unknown-linux-gnu -S --passes=expand-variadics --expand-variadics-override=optimize < %s | FileCheck %s -check-prefix=X64


declare void @llvm.va_start(ptr)
declare void @llvm.va_end(ptr)

declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture)

declare void @sink_valist(ptr)
declare void @sink_i32(i32)

%struct.__va_list_tag = type { i32, i32, ptr, ptr }

;; Simple function is split into two functions
; X86-LABEL: define internal void @x86_non_inlinable.valist(
; X86:       entry:
; X86:       %va = alloca ptr, align 4
; X86:       call void @sink_i32(i32 0)
; X86:       store ptr %varargs, ptr %va, align 4
; X86:       %0 = load ptr, ptr %va, align 4
; X86:       call void @sink_valist(ptr noundef %0)
; X86:       ret void
; X86:     }
; X86-LABEL: define void @x86_non_inlinable(
; X86:       entry:
; X86:       %va_list = alloca ptr, align 4
; X86:       call void @llvm.va_start.p0(ptr %va_list)
; X86:       tail call void @x86_non_inlinable.valist(ptr %va_list)
; X86:       ret void
; X86:       }
define void @x86_non_inlinable(...)  {
entry:
  %va = alloca ptr, align 4
  call void @sink_i32(i32 0)
  call void @llvm.va_start.p0(ptr nonnull %va)
  %0 = load ptr, ptr %va, align 4
  call void @sink_valist(ptr noundef %0)
  ret void
}

; TODO: This needs checks too
define void @x86_caller(i32 %x) {
  call void (...) @x86_non_inlinable(i32 %x)
  ret void
}


;; As above, but for x64 - the different va_list type means a missing load.
; X64-LABEL: define internal void @x64_non_inlinable.valist(
; X64:       entry:
; X64:       %va = alloca [1 x %struct.__va_list_tag], align 16
; X64:       call void @sink_i32(i32 0)
; X64:       call void @llvm.memcpy.inline.p0.p0.i32(ptr %va, ptr %varargs, i32 24, i1 false)
; X64:       call void @sink_valist(ptr noundef %va)
; X64:       ret void
; X64:     }
; X64-LABEL: define void @x64_non_inlinable(
; X64:       entry:
; X64:       %va_list = alloca [1 x { i32, i32, ptr, ptr }], align 8
; X64:       call void @llvm.va_start.p0(ptr %va_list)
; X64:       tail call void @x64_non_inlinable.valist(ptr %va_list)
; X64:       ret void
; X64:       }
define void @x64_non_inlinable(...)  {
entry:
  %va = alloca [1 x %struct.__va_list_tag], align 16
  call void @sink_i32(i32 0)
  call void @llvm.va_start.p0(ptr nonnull %va)
  call void @sink_valist(ptr noundef %va)
  ret void
}


; TODO: Is unchanged
define void @no_known_callers(...)  {
entry:
  %va = alloca ptr, align 4
  call void @sink_i32(i32 0)
  call void @llvm.va_start.p0(ptr nonnull %va)
  %0 = load ptr, ptr %va, align 4
  call void @sink_valist(ptr noundef %0)
  ret void
}

