; RUN: llvm-as < %s | llvm-dis | grep {tail call void.*sret null}

declare void @foo({  }* sret , ...)

define void @bar() {
        tail call void ({  }* sret , ...)* @foo( {  }* null sret , i32 0 )
        ret void
}
