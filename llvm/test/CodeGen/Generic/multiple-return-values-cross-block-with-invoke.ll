; RUN: llc < %s
declare { i64, double } @wild()

define void @foo(ptr %p, ptr %q) nounwind personality ptr @__gxx_personality_v0 {
        %t = invoke { i64, double } @wild() to label %normal unwind label %handler

normal:
        %mrv_gr = extractvalue { i64, double } %t, 0
        store i64 %mrv_gr, ptr %p
        %mrv_gr12681 = extractvalue { i64, double } %t, 1   
        store double %mrv_gr12681, ptr %q
	ret void
  
handler:
        %exn = landingpad {ptr, i32}
                 catch ptr null
	ret void
}

declare i32 @__gxx_personality_v0(...) addrspace(0)
