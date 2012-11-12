; RUN: llc < %s

declare { i64, double } @wild()

define void @foo(i64* %p, double* %q) nounwind {
        %t = invoke { i64, double } @wild() to label %normal unwind label %handler

normal:
        %mrv_gr = extractvalue { i64, double } %t, 0
        store i64 %mrv_gr, i64* %p
        %mrv_gr12681 = extractvalue { i64, double } %t, 1   
        store double %mrv_gr12681, double* %q
	ret void
  
handler:
        %exn = landingpad {i8*, i32} personality i32 (...)* @__gxx_personality_v0
                 catch i8* null
	ret void
}

declare i32 @__gxx_personality_v0(...)
