; RUN: llvm-as < %s | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll

define fastcc void @foo() {
        ret void
}

define coldcc void @bar() {
        call fastcc void @foo( )
        ret void
}

define void @structret({ i8 }* sret  %P) {
        call void @structret( { i8 }* sret  %P )
        ret void
}

define void @foo2() {
        ret void
}

define coldcc void @bar2() {
        call fastcc void @foo( )
        ret void
}

define cc42 void @bar3() {
        invoke fastcc void @foo( )
                        to label %Ok unwind label %U

Ok:             ; preds = %0
        ret void

U:              ; preds = %0
        unwind
}

define void @bar4() {
        call cc42 void @bar( )
        invoke cc42 void @bar3( )
                        to label %Ok unwind label %U

Ok:             ; preds = %0
        ret void

U:              ; preds = %0
        unwind
}

