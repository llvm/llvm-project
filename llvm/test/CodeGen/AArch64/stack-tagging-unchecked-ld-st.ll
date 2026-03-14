; RUN: llc < %s -mtriple=aarch64 -mattr=+mte | FileCheck %s --check-prefixes=DEFAULT,COMMON
; RUN: llc < %s -mtriple=aarch64 -mattr=+mte -stack-tagging-unchecked-ld-st=never | FileCheck %s --check-prefixes=NEVER,COMMON
; RUN: llc < %s -mtriple=aarch64 -mattr=+mte -stack-tagging-unchecked-ld-st=always | FileCheck %s --check-prefixes=ALWAYS,COMMON

declare void @use8(ptr)
declare void @use16(ptr)
declare void @use32(ptr)
declare void @use64(ptr)
declare void @use2x64(ptr)
declare void @llvm.lifetime.start.p0(i64, ptr nocapture)
declare void @llvm.lifetime.end.p0(i64, ptr nocapture)

define i64 @CallLd64() sanitize_memtag {
entry:
  %x = alloca i64, align 4
  call void @use64(ptr %x)
  %a = load i64, ptr %x
  ret i64 %a
}

; COMMON:  CallLd64:
; COMMON:  bl  use64

; ALWAYS:  ldr x0, [sp]
; DEFAULT: ldr x0, [sp]
; NEVER:   ldr x0, [x{{.*}}]

; COMMON:  ret


define i32 @CallLd32() sanitize_memtag {
entry:
  %x = alloca i32, align 4
  call void @use32(ptr %x)
  %a = load i32, ptr %x
  ret i32 %a
}

; COMMON:  CallLd32:
; COMMON:  bl  use32

; ALWAYS:  ldr w0, [sp]
; DEFAULT: ldr w0, [sp]
; NEVER:   ldr w0, [x{{.*}}]

; COMMON:  ret


define i16 @CallLd16() sanitize_memtag {
entry:
  %x = alloca i16, align 4
  call void @use16(ptr %x)
  %a = load i16, ptr %x
  ret i16 %a
}

; COMMON:  CallLd16:
; COMMON:  bl  use16

; ALWAYS:  ldrh w0, [sp]
; DEFAULT: ldrh w0, [sp]
; NEVER:   ldrh w0, [x{{.*}}]

; COMMON:  ret


define i8 @CallLd8() sanitize_memtag {
entry:
  %x = alloca i8, align 4
  call void @use8(ptr %x)
  %a = load i8, ptr %x
  ret i8 %a
}

; COMMON:  CallLd8:
; COMMON:  bl  use8

; ALWAYS:  ldrb w0, [sp]
; DEFAULT: ldrb w0, [sp]
; NEVER:   ldrb w0, [x{{.*}}]

; COMMON:  ret


define void @CallSt64Call() sanitize_memtag {
entry:
  %x = alloca i64, align 4
  call void @use64(ptr %x)
  store i64 42, ptr %x
  call void @use64(ptr %x)
  ret void
}

; COMMON:  CallSt64Call:
; COMMON:  bl  use64

; ALWAYS:  str x{{.*}}, [sp]
; DEFAULT: str x{{.*}}, [sp]
; NEVER:   str x{{.*}}, [x{{.*}}]

; COMMON:  bl  use64
; COMMON:  ret


define void @CallSt32Call() sanitize_memtag {
entry:
  %x = alloca i32, align 4
  call void @use32(ptr %x)
  store i32 42, ptr %x
  call void @use32(ptr %x)
  ret void
}

; COMMON:  CallSt32Call:
; COMMON:  bl  use32

; ALWAYS:  str w{{.*}}, [sp]
; DEFAULT: str w{{.*}}, [sp]
; NEVER:   str w{{.*}}, [x{{.*}}]

; COMMON:  bl  use32
; COMMON:  ret


define void @CallSt16Call() sanitize_memtag {
entry:
  %x = alloca i16, align 4
  call void @use16(ptr %x)
  store i16 42, ptr %x
  call void @use16(ptr %x)
  ret void
}


; COMMON:  CallSt16Call:
; COMMON:  bl  use16

; ALWAYS:  strh w{{.*}}, [sp]
; DEFAULT: strh w{{.*}}, [sp]
; NEVER:   strh w{{.*}}, [x{{.*}}]

; COMMON:  bl  use16
; COMMON:  ret


define void @CallSt8Call() sanitize_memtag {
entry:
  %x = alloca i8, align 4
  call void @use8(ptr %x)
  store i8 42, ptr %x
  call void @use8(ptr %x)
  ret void
}

; COMMON:  CallSt8Call:
; COMMON:  bl  use8

; ALWAYS:  strb w{{.*}}, [sp]
; DEFAULT: strb w{{.*}}, [sp]
; NEVER:   strb w{{.*}}, [x{{.*}}]

; COMMON:  bl  use8
; COMMON:  ret


define void @CallStPair(i64 %z) sanitize_memtag {
entry:
  %x = alloca [2 x i64], align 8
  call void @use2x64(ptr %x)
  store i64 %z, ptr %x, align 8
  %x1 = getelementptr inbounds [2 x i64], ptr %x, i64 0, i64 1
  store i64 %z, ptr %x1, align 8
  call void @use2x64(ptr %x)
  ret void
}

; COMMON:  CallStPair:
; COMMON:  bl  use2x64

; ALWAYS:  stp {{.*}}, [sp]
; DEFAULT: stp {{.*}}, [sp]
; NEVER:   stp {{.*}}, [x{{.*}}]

; COMMON:  bl  use2x64
; COMMON:  ret

; One of the two allocas will end up out of range of ldrb [sp].
define dso_local i8 @LargeFrame() sanitize_memtag {
entry:
  %x = alloca [4096 x i8], align 4
  %y = alloca [4096 x i8], align 4
  call void @use8(ptr %x)
  call void @use8(ptr %y)
  %0 = load i8, ptr %x, align 4
  %1 = load i8, ptr %y, align 4
  %add = add i8 %1, %0
  ret i8 %add
}

; COMMON: LargeFrame:
; COMMON: bl use8
; COMMON: bl use8

; NEVER:  ldrb [[A:w.*]], [x{{.*}}]
; NEVER:  ldrb [[B:w.*]], [x{{.*}}]

; DEFAULT:  ldrb [[A:w.*]], [x{{.*}}]
; DEFAULT:  ldrb [[B:w.*]], [x{{.*}}]

; ALWAYS-DAG: ldg [[PA:x.*]], [x{{.*}}]
; ALWAYS-DAG: ldrb [[B:w.*]], [sp]
; ALWAYS-DAG: ldrb [[A:w.*]], [[[PA]]]

; COMMON: ret

; One of these allocas is closer to FP than to SP, and within 256 bytes
; of the former (see hardcoded limit in resolveFrameOffsetReference).
; It could be lowered to an FP-relative load, but not when doing an
; unchecked access to tagged memory!
define i8 @FPOffset() "frame-pointer"="all" sanitize_memtag {
  %x = alloca [200 x i8], align 4
  %y = alloca [200 x i8], align 4
  %z = alloca [200 x i8], align 4
  call void @use8(ptr %x)
  call void @use8(ptr %y)
  call void @use8(ptr %z)
  %x1 = load i8, ptr %x, align 4
  %y1 = load i8, ptr %y, align 4
  %z1 = load i8, ptr %z, align 4
  %a = add i8 %x1, %y1
  %b = add i8 %a, %z1
  ret i8 %b
}

; COMMON: FPOffset:
; COMMON: bl use8
; COMMON: bl use8
; COMMON: bl use8

; All three loads are SP-based.
; ALWAYS-DAG: ldrb  w{{.*}}, [sp, #416]
; ALWAYS-DAG: ldrb  w{{.*}}, [sp, #208]
; ALWAYS-DAG: ldrb  w{{.*}}, [sp]

; DEFAULT-DAG: ldrb  w{{.*}}, [sp, #416]
; DEFAULT-DAG: ldrb  w{{.*}}, [sp, #208]
; DEFAULT-DAG: ldrb  w{{.*}}, [sp]

; NEVER-DAG: ldrb  w{{.*}}, [x{{.*}}]
; NEVER-DAG: ldrb  w{{.*}}, [x{{.*}}]
; NEVER-DAG: ldrb  w{{.*}}, [x{{.*}}]

; COMMON: ret
