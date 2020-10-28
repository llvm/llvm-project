; RUN: llc -fast-isel-sink-local-values < %s -O0 -fast-isel-abort=1 -relocation-model=dynamic-no-pic -mtriple=armv7-apple-ios -verify-machineinstrs | FileCheck %s --check-prefix=ARM --check-prefix=ARM-MACHO
; RUN: llc -fast-isel-sink-local-values < %s -O0 -fast-isel-abort=1 -relocation-model=dynamic-no-pic -mtriple=armv7-linux-gnueabi -verify-machineinstrs | FileCheck %s --check-prefix=ARM --check-prefix=ARM-ELF
; RUN: llc -fast-isel-sink-local-values < %s -O0 -fast-isel-abort=1 -relocation-model=dynamic-no-pic -mtriple=thumbv7-apple-ios -verify-machineinstrs | FileCheck %s --check-prefix=THUMB
; RUN: llc -fast-isel-sink-local-values < %s -O0 -fast-isel-abort=1 -relocation-model=dynamic-no-pic -mtriple=armv7-apple-ios -mattr=+long-calls -verify-machineinstrs | FileCheck %s --check-prefix=ARM-LONG --check-prefix=ARM-LONG-MACHO
; RUN: llc -fast-isel-sink-local-values < %s -O0 -fast-isel-abort=1 -relocation-model=dynamic-no-pic -mtriple=armv7-linux-gnueabi -mattr=+long-calls -verify-machineinstrs | FileCheck %s --check-prefix=ARM-LONG --check-prefix=ARM-LONG-ELF
; RUN: llc -fast-isel-sink-local-values < %s -O0 -fast-isel-abort=1 -relocation-model=dynamic-no-pic -mtriple=thumbv7-apple-ios -mattr=+long-calls -verify-machineinstrs | FileCheck %s --check-prefix=THUMB-LONG

; Note that some of these tests assume that relocations are either
; movw/movt or constant pool loads. Different platforms will select
; different approaches.

@message1 = global [60 x i8] c"The LLVM Compiler Infrastructure\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00", align 1
@temp = common global [60 x i8] zeroinitializer, align 1

define void @t1() nounwind ssp {
; ARM-LABEL: t1:
; ARM: {{(movw r0, :lower16:_?message1)|(ldr r0, .LCPI)}}
; ARM: {{(movt r0, :upper16:_?message1)|(ldr r0, \[r0\])}}
; ARM-DAG: add r0, r0, #5
; ARM-DAG: movw r1, #64
; ARM-DAG: movw r2, #10
; ARM-DAG: and r1, r1, #255
; ARM: bl {{_?}}memset
; ARM-LONG-LABEL: t1:

; ARM-LONG-MACHO: {{(movw r3, :lower16:L_memset\$non_lazy_ptr)|(ldr r3, .LCPI)}}
; ARM-LONG-MACHO: {{(movt r3, :upper16:L_memset\$non_lazy_ptr)?}}
; ARM-LONG-MACHO: ldr r3, [r3]

; ARM-LONG-ELF: movw r3, :lower16:memset
; ARM-LONG-ELF: movt r3, :upper16:memset

; ARM-LONG: blx r3
; THUMB-LABEL: t1:
; THUMB: {{(movw r0, :lower16:_?message1)|(ldr.n r0, .LCPI)}}
; THUMB: {{(movt r0, :upper16:_?message1)|(ldr r0, \[r0\])}}
; THUMB: adds r0, #5
; THUMB: movs r1, #64
; THUMB: and r1, r1, #255
; THUMB: movs r2, #10
; THUMB: bl {{_?}}memset
; THUMB-LONG-LABEL: t1:
; THUMB-LONG: movw r3, :lower16:L_memset$non_lazy_ptr
; THUMB-LONG: movt r3, :upper16:L_memset$non_lazy_ptr
; THUMB-LONG: ldr r3, [r3]
; THUMB-LONG: blx r3
  call void @llvm.memset.p0i8.i32(i8* align 4 getelementptr inbounds ([60 x i8], [60 x i8]* @message1, i32 0, i32 5), i8 64, i32 10, i1 false)
  ret void
}

declare void @llvm.memset.p0i8.i32(i8* nocapture, i8, i32, i1) nounwind

define void @t2() nounwind ssp {
; ARM-LABEL: t2:

; ARM-MACHO: {{(movw r0, :lower16:L_temp\$non_lazy_ptr)|(ldr r0, .LCPI)}}
; ARM-MACHO: {{(movt r0, :upper16:L_temp\$non_lazy_ptr)?}}
; ARM-MACHO: ldr [[REG1:r[0-9]+]], [r0]

; ARM-ELF: movw [[REG1:r[0-9]+]], :lower16:temp
; ARM-ELF: movt [[REG1]], :upper16:temp

; ARM: add r0, [[REG1]], #4
; ARM: add r1, [[REG1]], #16
; ARM: movw r2, #17
; ARM: bl {{_?}}memcpy
; ARM-LONG-LABEL: t2:

; ARM-LONG-MACHO: {{(movw r3, :lower16:L_memcpy\$non_lazy_ptr)|(ldr r3, .LCPI)}}
; ARM-LONG-MACHO: {{(movt r3, :upper16:L_memcpy\$non_lazy_ptr)?}}
; ARM-LONG-MACHO: ldr r3, [r3]

; ARM-LONG-ELF: movw r3, :lower16:memcpy
; ARM-LONG-ELF: movt r3, :upper16:memcpy

; ARM-LONG: blx r3
; THUMB-LABEL: t2:
; THUMB: {{(movw r0, :lower16:L_temp\$non_lazy_ptr)|(ldr.n r0, .LCPI)}}
; THUMB: {{(movt r0, :upper16:L_temp\$non_lazy_ptr)?}}
; THUMB: ldr [[REG1:r[0-9]+]], [r0]
; THUMB: adds r0, [[REG1]], #4
; THUMB: adds r1, #16
; THUMB: movs r2, #17
; THUMB: bl {{_?}}memcpy
; THUMB-LONG-LABEL: t2:
; THUMB-LONG: movw r3, :lower16:L_memcpy$non_lazy_ptr
; THUMB-LONG: movt r3, :upper16:L_memcpy$non_lazy_ptr
; THUMB-LONG: ldr r3, [r3]
; THUMB-LONG: blx r3
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 4 getelementptr inbounds ([60 x i8], [60 x i8]* @temp, i32 0, i32 4), i8* align 4 getelementptr inbounds ([60 x i8], [60 x i8]* @temp, i32 0, i32 16), i32 17, i1 false)
  ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i1) nounwind

define void @t3() nounwind ssp {
; ARM-LABEL: t3:

; ARM-MACHO: {{(movw r0, :lower16:L_temp\$non_lazy_ptr)|(ldr r0, .LCPI)}}
; ARM-MACHO: {{(movt r0, :upper16:L_temp\$non_lazy_ptr)?}}
; ARM-MACHO: ldr [[REG0:r[0-9]+]], [r0]

; ARM-ELF: movw [[REG0:r[0-9]+]], :lower16:temp
; ARM-ELF: movt [[REG0]], :upper16:temp


; ARM: add r0, [[REG0]], #4
; ARM: add r1, [[REG0]], #16
; ARM: movw r2, #10
; ARM: bl {{_?}}memmove
; ARM-LONG-LABEL: t3:

; ARM-LONG-MACHO: {{(movw r3, :lower16:L_memmove\$non_lazy_ptr)|(ldr r3, .LCPI)}}
; ARM-LONG-MACHO: {{(movt r3, :upper16:L_memmove\$non_lazy_ptr)?}}
; ARM-LONG-MACHO: ldr r3, [r3]

; ARM-LONG-ELF: movw r3, :lower16:memmove
; ARM-LONG-ELF: movt r3, :upper16:memmove

; ARM-LONG: blx r3
; THUMB-LABEL: t3:
; THUMB: {{(movw r0, :lower16:L_temp\$non_lazy_ptr)|(ldr.n r0, .LCPI)}}
; THUMB: {{(movt r0, :upper16:L_temp\$non_lazy_ptr)?}}
; THUMB: ldr [[REG1:r[0-9]+]], [r0]
; THUMB: adds r0, [[REG1]], #4
; THUMB: adds r1, #16
; THUMB: movs r2, #10
; THUMB: bl {{_?}}memmove
; THUMB-LONG-LABEL: t3:
; THUMB-LONG: movw r3, :lower16:L_memmove$non_lazy_ptr
; THUMB-LONG: movt r3, :upper16:L_memmove$non_lazy_ptr
; THUMB-LONG: ldr r3, [r3]
; THUMB-LONG: blx r3
  call void @llvm.memmove.p0i8.p0i8.i32(i8* align 1 getelementptr inbounds ([60 x i8], [60 x i8]* @temp, i32 0, i32 4), i8* align 1 getelementptr inbounds ([60 x i8], [60 x i8]* @temp, i32 0, i32 16), i32 10, i1 false)
  ret void
}

define void @t4() nounwind ssp {
; ARM-LABEL: t4:

; ARM-MACHO: {{(movw r0, :lower16:L_temp\$non_lazy_ptr)|(ldr r0, .LCPI)}}
; ARM-MACHO: {{(movt r0, :upper16:L_temp\$non_lazy_ptr)?}}
; ARM-MACHO: ldr [[REG0:r[0-9]+]], [r0]

; ARM-ELF: movw [[REG0:r[0-9]+]], :lower16:temp
; ARM-ELF: movt [[REG0]], :upper16:temp

; ARM: ldr [[REG1:r[0-9]+]], {{\[}}[[REG0]], #16]
; ARM: str [[REG1]], {{\[}}[[REG0]], #4]
; ARM: ldr [[REG2:r[0-9]+]], {{\[}}[[REG0]], #20]
; ARM: str [[REG2]], {{\[}}[[REG0]], #8]
; ARM: ldrh [[REG3:r[0-9]+]], {{\[}}[[REG0]], #24]
; ARM: strh [[REG3]], {{\[}}[[REG0]], #12]
; ARM: bx lr
; THUMB-LABEL: t4:
; THUMB: {{(movw r0, :lower16:L_temp\$non_lazy_ptr)|(ldr.n r0, .LCPI)}}
; THUMB: {{(movt r0, :upper16:L_temp\$non_lazy_ptr)?}}
; THUMB: ldr [[REG1:r[0-9]+]], [r0]
; THUMB: ldr [[REG2:r[0-9]+]], {{\[}}[[REG1]], #16]
; THUMB: str [[REG2]], {{\[}}[[REG1]], #4]
; THUMB: ldr [[REG3:r[0-9]+]], {{\[}}[[REG1]], #20]
; THUMB: str [[REG3]], {{\[}}[[REG1]], #8]
; THUMB: ldrh [[REG4:r[0-9]+]], {{\[}}[[REG1]], #24]
; THUMB: strh [[REG4]], {{\[}}[[REG1]], #12]
; THUMB: bx lr
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 4 getelementptr inbounds ([60 x i8], [60 x i8]* @temp, i32 0, i32 4), i8* align 4 getelementptr inbounds ([60 x i8], [60 x i8]* @temp, i32 0, i32 16), i32 10, i1 false)
  ret void
}

declare void @llvm.memmove.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i1) nounwind

define void @t5() nounwind ssp {
; ARM-LABEL: t5:

; ARM-MACHO: {{(movw r0, :lower16:L_temp\$non_lazy_ptr)|(ldr r0, .LCPI)}}
; ARM-MACHO: {{(movt r0, :upper16:L_temp\$non_lazy_ptr)?}}
; ARM-MACHO: ldr [[REG0:r[0-9]+]], [r0]

; ARM-ELF: movw [[REG0:r[0-9]+]], :lower16:temp
; ARM-ELF: movt [[REG0]], :upper16:temp

; ARM: ldrh [[REG1:r[0-9]+]], {{\[}}[[REG0]], #16]
; ARM: strh [[REG1]], {{\[}}[[REG0]], #4]
; ARM: ldrh [[REG2:r[0-9]+]], {{\[}}[[REG0]], #18]
; ARM: strh [[REG2]], {{\[}}[[REG0]], #6]
; ARM: ldrh [[REG3:r[0-9]+]], {{\[}}[[REG0]], #20]
; ARM: strh [[REG3]], {{\[}}[[REG0]], #8]
; ARM: ldrh [[REG4:r[0-9]+]], {{\[}}[[REG0]], #22]
; ARM: strh [[REG4]], {{\[}}[[REG0]], #10]
; ARM: ldrh [[REG5:r[0-9]+]], {{\[}}[[REG0]], #24]
; ARM: strh [[REG5]], {{\[}}[[REG0]], #12]
; ARM: bx lr
; THUMB-LABEL: t5:
; THUMB: {{(movw r0, :lower16:L_temp\$non_lazy_ptr)|(ldr.n r0, .LCPI)}}
; THUMB: {{(movt r0, :upper16:L_temp\$non_lazy_ptr)?}}
; THUMB: ldr [[REG1:r[0-9]+]], [r0]
; THUMB: ldrh [[REG2:r[0-9]+]], {{\[}}[[REG1]], #16]
; THUMB: strh [[REG2]], {{\[}}[[REG1]], #4]
; THUMB: ldrh [[REG3:r[0-9]+]], {{\[}}[[REG1]], #18]
; THUMB: strh [[REG3]], {{\[}}[[REG1]], #6]
; THUMB: ldrh [[REG4:r[0-9]+]], {{\[}}[[REG1]], #20]
; THUMB: strh [[REG4]], {{\[}}[[REG1]], #8]
; THUMB: ldrh [[REG5:r[0-9]+]], {{\[}}[[REG1]], #22]
; THUMB: strh [[REG5]], {{\[}}[[REG1]], #10]
; THUMB: ldrh [[REG6:r[0-9]+]], {{\[}}[[REG1]], #24]
; THUMB: strh [[REG6]], {{\[}}[[REG1]], #12]
; THUMB: bx lr
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 2 getelementptr inbounds ([60 x i8], [60 x i8]* @temp, i32 0, i32 4), i8* align 2 getelementptr inbounds ([60 x i8], [60 x i8]* @temp, i32 0, i32 16), i32 10, i1 false)
  ret void
}

define void @t6() nounwind ssp {
; ARM-LABEL: t6:

; ARM-MACHO: {{(movw r0, :lower16:L_temp\$non_lazy_ptr)|(ldr r0, .LCPI)}}
; ARM-MACHO: {{(movt r0, :upper16:L_temp\$non_lazy_ptr)?}}
; ARM-MACHO: ldr [[REG0:r[0-9]+]], [r0]

; ARM-ELF: movw [[REG0:r[0-9]+]], :lower16:temp
; ARM-ELF: movt [[REG0]], :upper16:temp

; ARM: ldrb [[REG1:r[0-9]+]], {{\[}}[[REG0]], #16]
; ARM: strb [[REG1]], {{\[}}[[REG0]], #4]
; ARM: ldrb [[REG2:r[0-9]+]], {{\[}}[[REG0]], #17]
; ARM: strb [[REG2]], {{\[}}[[REG0]], #5]
; ARM: ldrb [[REG3:r[0-9]+]], {{\[}}[[REG0]], #18]
; ARM: strb [[REG3]], {{\[}}[[REG0]], #6]
; ARM: ldrb [[REG4:r[0-9]+]], {{\[}}[[REG0]], #19]
; ARM: strb [[REG4]], {{\[}}[[REG0]], #7]
; ARM: ldrb [[REG5:r[0-9]+]], {{\[}}[[REG0]], #20]
; ARM: strb [[REG5]], {{\[}}[[REG0]], #8]
; ARM: ldrb [[REG6:r[0-9]+]], {{\[}}[[REG0]], #21]
; ARM: strb [[REG6]], {{\[}}[[REG0]], #9]
; ARM: ldrb [[REG7:r[0-9]+]], {{\[}}[[REG0]], #22]
; ARM: strb [[REG7]], {{\[}}[[REG0]], #10]
; ARM: ldrb [[REG8:r[0-9]+]], {{\[}}[[REG0]], #23]
; ARM: strb [[REG8]], {{\[}}[[REG0]], #11]
; ARM: ldrb [[REG9:r[0-9]+]], {{\[}}[[REG0]], #24]
; ARM: strb [[REG9]], {{\[}}[[REG0]], #12]
; ARM: ldrb [[REG10:r[0-9]+]], {{\[}}[[REG0]], #25]
; ARM: strb [[REG10]], {{\[}}[[REG0]], #13]
; ARM: bx lr
; THUMB-LABEL: t6:
; THUMB: {{(movw r0, :lower16:L_temp\$non_lazy_ptr)|(ldr.n r0, .LCPI)}}
; THUMB: {{(movt r0, :upper16:L_temp\$non_lazy_ptr)?}}
; THUMB: ldr [[REG0:r[0-9]+]], [r0]
; THUMB: ldrb [[REG2:r[0-9]+]], {{\[}}[[REG0]], #16]
; THUMB: strb [[REG2]], {{\[}}[[REG0]], #4]
; THUMB: ldrb [[REG3:r[0-9]+]], {{\[}}[[REG0]], #17]
; THUMB: strb [[REG3]], {{\[}}[[REG0]], #5]
; THUMB: ldrb [[REG4:r[0-9]+]], {{\[}}[[REG0]], #18]
; THUMB: strb [[REG4]], {{\[}}[[REG0]], #6]
; THUMB: ldrb [[REG5:r[0-9]+]], {{\[}}[[REG0]], #19]
; THUMB: strb [[REG5]], {{\[}}[[REG0]], #7]
; THUMB: ldrb [[REG6:r[0-9]+]], {{\[}}[[REG0]], #20]
; THUMB: strb [[REG6]], {{\[}}[[REG0]], #8]
; THUMB: ldrb [[REG7:r[0-9]+]], {{\[}}[[REG0]], #21]
; THUMB: strb [[REG7]], {{\[}}[[REG0]], #9]
; THUMB: ldrb [[REG8:r[0-9]+]], {{\[}}[[REG0]], #22]
; THUMB: strb [[REG8]], {{\[}}[[REG0]], #10]
; THUMB: ldrb [[REG9:r[0-9]+]], {{\[}}[[REG0]], #23]
; THUMB: strb [[REG9]], {{\[}}[[REG0]], #11]
; THUMB: ldrb [[REG10:r[0-9]+]], {{\[}}[[REG0]], #24]
; THUMB: strb [[REG10]], {{\[}}[[REG0]], #12]
; THUMB: ldrb [[REG11:r[0-9]+]], {{\[}}[[REG0]], #25]
; THUMB: strb [[REG11]], {{\[}}[[REG0]], #13]
; THUMB: bx lr
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 1 getelementptr inbounds ([60 x i8], [60 x i8]* @temp, i32 0, i32 4), i8* align 1 getelementptr inbounds ([60 x i8], [60 x i8]* @temp, i32 0, i32 16), i32 10, i1 false)
  ret void
}

; rdar://13202135
define void @t7() nounwind ssp {
; Just make sure this doesn't assert when we have an odd length and an alignment of 2.
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 2 getelementptr inbounds ([60 x i8], [60 x i8]* @temp, i32 0, i32 4), i8* align 2 getelementptr inbounds ([60 x i8], [60 x i8]* @temp, i32 0, i32 16), i32 3, i1 false)
  ret void
}

define i32 @t8(i32 %x) nounwind {
entry:
; ARM-LABEL: t8:
; ARM-NOT: FastISel missed call:   %expval = call i32 @llvm.expect.i32(i32 %x, i32 1)
; THUMB-LABEL: t8:
; THUMB-NOT: FastISel missed call:   %expval = call i32 @llvm.expect.i32(i32 %x, i32 1)
  %expval = call i32 @llvm.expect.i32(i32 %x, i32 1)
  ret i32 %expval
}

declare i32 @llvm.expect.i32(i32, i32) nounwind readnone
