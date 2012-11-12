; RUN: opt < %s -inline -loop-rotate -verify-dom-info -verify-loop-info -disable-output
; PR3601
declare void @solve()

define internal fastcc void @read() {
	br label %bb4

bb3:
	br label %bb4

bb4:
	call void @solve()
	br i1 false, label %bb5, label %bb3

bb5:
	unreachable
}

define internal fastcc void @parse() {
	call fastcc void @read()
	ret void
}

define void @main() {
	invoke fastcc void @parse()
			to label %invcont unwind label %lpad

invcont:
	unreachable

lpad:
        %exn = landingpad {i8*, i32} personality i32 (...)* @__gxx_personality_v0
                 cleanup
	unreachable
}
declare i32 @__gxx_personality_v0(...)
