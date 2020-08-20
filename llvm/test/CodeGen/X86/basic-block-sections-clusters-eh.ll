; BB cluster section test for exception handling.
;
; Test1: Basic blocks #1 and #3 are landing pads and must be in the same section.
; Basic block 2 will be placed in a unique section, but #1 and #3 are placed in the special exception section.
; The rest will be placed in a section along with the entry basic block.
; RUN: echo '!main' > %t1
; RUN: echo '!!1 2' >> %t1
; RUN: llc < %s -O0 -mtriple=x86_64-pc-linux -function-sections -basic-block-sections=%t1 | FileCheck %s -check-prefix=LINUX-SECTIONS1
;
; Test2: Basic blocks #1, #2, and #3 go into a separate section.
; No separate exception section will be created as #1 and #3 are already in one section.
; The rest will be placed in a section along with the entry basic block.
; RUN: echo '!main' > %t2
; RUN: echo '!!1 2 3' >> %t2
; RUN: llc < %s -O0 -mtriple=x86_64-pc-linux -function-sections -basic-block-sections=%t2 | FileCheck %s -check-prefix=LINUX-SECTIONS2

@_ZTIi = external constant i8*

define i32 @main() uwtable optsize ssp personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  invoke void @_Z1fv() optsize
          to label %try.cont unwind label %lpad1

lpad1:
  %0 = landingpad { i8*, i32 }
          cleanup
          catch i8* bitcast (i8** @_ZTIi to i8*)
  br label %eh.resume1

try.cont:
  invoke void @_Z2fv() optsize
          to label %try.cont unwind label %lpad2
  ret i32 0

lpad2:
  %2 = landingpad { i8*, i32 }
          cleanup
          catch i8* bitcast (i8** @_ZTIi to i8*)
  br label %eh.resume2

eh.resume1:
  resume { i8*, i32 } %0

eh.resume2:
  resume { i8*, i32 } %2
}

declare void @_Z1fv() optsize

declare void @_Z2fv() optsize

declare i32 @__gxx_personality_v0(...)

; LINUX-SECTIONS1:		.section	.text.main,"ax",@progbits
; LINUX-SECTIONS1-LABEL:	main:
; LINUX-SECTIONS1-NOT: 		.section
; LINUX-SECTIONS1-LABEL:	.LBB0_4:
; LINUX-SECTIONS1-NOT: 		.section
; LINUX-SECTIONS1-LABEL:	.LBB0_5:
; LINUX-SECTIONS1-NOT: 		.section
; LINUX-SECTIONS1-LABEL:	.LBB0_6:
; LINUX-SECTIONS1: 		.section	.text.main,"ax",@progbits,unique,1
; LINUX-SECTIONS1-LABEL:	main.0:
; LINUX-SECTIONS1:		.section	.text.eh.main,"ax",@progbits
; LINUX-SECTIONS1-LABEL: 	main.eh:
; LINUX-SECTIONS1-NOT: 		.section
; LINUX-SECTIONS1-LABEL:	.LBB0_3:
; LINUX-SECTIONS1-NOT:		.section
; LINUX-SECTIONS1:		.section	.text.main,"ax",@progbits
; LINUX-SECTIONS1-LABEL: 	.Lfunc_end0


; LINUX-SECTIONS2:		.section	.text.main,"ax",@progbits
; LINUX-SECTIONS2-LABEL:	main:
; LINUX-SECTIONS2-NOT: 		.section
; LINUX-SECTIONS2-LABEL:	.LBB0_4:
; LINUX-SECTIONS2-NOT: 		.section
; LINUX-SECTIONS2-LABEL:	.LBB0_5:
; LINUX-SECTIONS2-NOT: 		.section
; LINUX-SECTIONS2-LABEL:	.LBB0_6:
; LINUX-SECTIONS2: 		.section	.text.main,"ax",@progbits,unique,1
; LINUX-SECTIONS2-LABEL: 	main.0:
; LINUX-SECTIONS2-NOT: 		.section
; LINUX-SECTIONS2-LABEL:	.LBB0_2:
; LINUX-SECTIONS2-NOT: 		.section
; LINUX-SECTIONS2-LABEL:	.LBB0_3:
; LINUX-SECTIONS2:		.section	.text.main,"ax",@progbits
; LINUX-SECTIONS2-LABEL: 	.Lfunc_end0
