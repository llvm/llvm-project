; RUN:  opt -S --passes=ipsccp,deadargelim --force-specialization                       < %s | FileCheck %s --check-prefix=NO-GLOBALS
; RUN:  opt -S --passes=ipsccp,deadargelim --force-specialization --funcspec-on-address < %s | FileCheck %s --check-prefix=GLOBALS
@G = global [10 x i32] zeroinitializer, align 4

define internal i32 @f(ptr %p) noinline {
entry:
  %0 = load i32, ptr %p, align 4
  store i32 0, ptr %p, align 4
  ret i32 %0
}

define internal i32 @g(i32 %x, i32 %y, ptr %p) noinline {
entry:
  %cmp = icmp sgt i32 %x, %y
  br i1 %cmp, label %if.then, label %if.else

if.then:
  br label %if.end

if.else:
  br label %if.end

if.end:
  %x.addr.0 = phi i32 [ %x, %if.then ], [ 11, %if.else ]
  %p.addr.0 = phi ptr [ @G, %if.then ], [ %p, %if.else ]
  %call = call i32 @f(ptr %p.addr.0)
  %add = add nsw i32 %call, %x.addr.0
  ret i32 %add
}

define i32 @h0(ptr %p) {
entry:
  %call = call i32 @g(i32 2, i32 1, ptr %p)
  ret i32 %call
}

define i32 @h1() {
entry:
  %call = call i32 @f(ptr @G)
  ret i32 %call
}

define i32 @h2() {
entry:
  %call = call i32 @f(ptr getelementptr inbounds (i32, ptr @G, i64 1))
  ret i32 %call
}

; Check if specialisation on the address of a non-const global variable
; is not allowed, then it is not performed.

; NO-GLOBALS-LABEL: define internal range(i32 -2147483646, -2147483648) i32 @g()
; NO-GLOBALS: call i32 @f(ptr @G)

; NO-GLOBALS-LABEL: define range(i32 -2147483646, -2147483648) i32 @h0(ptr %p)
; NO-GLOBALS:call i32 @g()

; NO-GLOBALS-LABEL: define i32 @h1()
; NO-GLOBALS: call i32 @f(ptr @G)

; NO-GLOBALS-LABEL: define i32 @h2()
; NO-GLOBALS: call i32 @f(ptr getelementptr inbounds (i32, ptr @G, i64 1))

; Check if specialisation on the address of a non-const global variable
; is allowed, then it is performed where possible.

; GLOBALS-LABEL: define internal range(i32 -2147483646, -2147483648) i32 @g()
; GLOBALS: call i32 @f.specialized.2()

; GLOBALS-LABEL: define range(i32 -2147483646, -2147483648) i32 @h0(ptr %p)
; GLOBALS: call i32 @g()

; GLOBALS-LABEL: define i32 @h1()
; GLOBALS: call i32 @f.specialized.2()

; GLOBALS-LABEL: define i32 @h2()
; GLOBALS: call i32 @f.specialized.1()

