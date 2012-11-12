; RUN: llc < %s -march=thumb -mattr=+thumb2 -arm-adjust-jump-tables=0 | FileCheck %s

; Do not use tbb / tbh if any destination is before the jumptable.
; rdar://7102917

define i16 @main__getopt_internal_2E_exit_2E_ce(i32, i1 %b) nounwind {
; CHECK: main__getopt_internal_2E_exit_2E_ce
; CHECK-NOT: tbb
; CHECK-NOT: tbh
; 32-bit jump tables use explicit branches, not data regions, so make sure
; we don't annotate this region.
; CHECK-NOT: data_region
entry:
  br i1 %b, label %codeRepl127.exitStub, label %newFuncRoot

newFuncRoot:
	br label %_getopt_internal.exit.ce

codeRepl127.exitStub:		; preds = %_getopt_internal.exit.ce
  ; Add an explicit edge back to before the jump table to ensure this block
  ; is placed first.
  br i1 %b, label %newFuncRoot, label %codeRepl127.exitStub.exit

codeRepl127.exitStub.exit:
	ret i16 0

parse_options.exit.loopexit.exitStub:		; preds = %_getopt_internal.exit.ce
	ret i16 1

bb1.i.exitStub:		; preds = %_getopt_internal.exit.ce
	ret i16 2

bb90.i.exitStub:		; preds = %_getopt_internal.exit.ce
	ret i16 3

codeRepl104.exitStub:		; preds = %_getopt_internal.exit.ce
	ret i16 4

codeRepl113.exitStub:		; preds = %_getopt_internal.exit.ce
	ret i16 5

codeRepl51.exitStub:		; preds = %_getopt_internal.exit.ce
	ret i16 6

codeRepl70.exitStub:		; preds = %_getopt_internal.exit.ce
	ret i16 7

codeRepl119.exitStub:		; preds = %_getopt_internal.exit.ce
	ret i16 8

codeRepl93.exitStub:		; preds = %_getopt_internal.exit.ce
	ret i16 9

codeRepl101.exitStub:		; preds = %_getopt_internal.exit.ce
	ret i16 10

codeRepl120.exitStub:		; preds = %_getopt_internal.exit.ce
	ret i16 11

codeRepl89.exitStub:		; preds = %_getopt_internal.exit.ce
	ret i16 12

codeRepl45.exitStub:		; preds = %_getopt_internal.exit.ce
	ret i16 13

codeRepl58.exitStub:		; preds = %_getopt_internal.exit.ce
	ret i16 14

codeRepl46.exitStub:		; preds = %_getopt_internal.exit.ce
	ret i16 15

codeRepl50.exitStub:		; preds = %_getopt_internal.exit.ce
	ret i16 16

codeRepl52.exitStub:		; preds = %_getopt_internal.exit.ce
	ret i16 17

codeRepl53.exitStub:		; preds = %_getopt_internal.exit.ce
	ret i16 18

codeRepl61.exitStub:		; preds = %_getopt_internal.exit.ce
	ret i16 19

codeRepl85.exitStub:		; preds = %_getopt_internal.exit.ce
	ret i16 20

codeRepl97.exitStub:		; preds = %_getopt_internal.exit.ce
	ret i16 21

codeRepl79.exitStub:		; preds = %_getopt_internal.exit.ce
	ret i16 22

codeRepl102.exitStub:		; preds = %_getopt_internal.exit.ce
	ret i16 23

codeRepl54.exitStub:		; preds = %_getopt_internal.exit.ce
	ret i16 24

codeRepl57.exitStub:		; preds = %_getopt_internal.exit.ce
	ret i16 25

codeRepl103.exitStub:		; preds = %_getopt_internal.exit.ce
	ret i16 26

_getopt_internal.exit.ce:		; preds = %newFuncRoot
	switch i32 %0, label %codeRepl127.exitStub [
		i32 -1, label %parse_options.exit.loopexit.exitStub
		i32 0, label %bb1.i.exitStub
		i32 63, label %bb90.i.exitStub
		i32 66, label %codeRepl104.exitStub
		i32 67, label %codeRepl113.exitStub
		i32 71, label %codeRepl51.exitStub
		i32 77, label %codeRepl70.exitStub
		i32 78, label %codeRepl119.exitStub
		i32 80, label %codeRepl93.exitStub
		i32 81, label %codeRepl101.exitStub
		i32 82, label %codeRepl120.exitStub
		i32 88, label %codeRepl89.exitStub
		i32 97, label %codeRepl45.exitStub
		i32 98, label %codeRepl58.exitStub
		i32 99, label %codeRepl46.exitStub
		i32 100, label %codeRepl50.exitStub
		i32 104, label %codeRepl52.exitStub
		i32 108, label %codeRepl53.exitStub
		i32 109, label %codeRepl61.exitStub
		i32 110, label %codeRepl85.exitStub
		i32 111, label %codeRepl97.exitStub
		i32 113, label %codeRepl79.exitStub
		i32 114, label %codeRepl102.exitStub
		i32 115, label %codeRepl54.exitStub
		i32 116, label %codeRepl57.exitStub
		i32 118, label %codeRepl103.exitStub
	]
}
