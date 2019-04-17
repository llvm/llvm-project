; RUN: opt < %s -inline -disable-output

define i32 @main() personality i32 (...)* @__gxx_personality_v0 {
entry:
        invoke void @__main( )
                        to label %LongJmpBlkPost unwind label %LongJmpBlkPre

LongJmpBlkPost:
        ret i32 0

LongJmpBlkPre:
        %i.3 = phi i32 [ 0, %entry ]
        %exn = landingpad {i8*, i32}
                 cleanup
        ret i32 0
}

define void @__main() {
        ret void
}

declare i32 @__gxx_personality_v0(...)
