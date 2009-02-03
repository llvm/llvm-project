; RUN: llvm-as < %s | llc -march=alpha

target datalayout = "e-p:64:64"
target triple = "alphaev67-unknown-linux-gnu"
        %llvm.dbg.compile_unit.type = type { i32, {  }*, i32, i32, i8*, i8*, i8* }
        %struct._Callback_list = type { %struct._Callback_list*, void (i32, %struct.ios_base*, i32)*, i32, i32 }
        %struct._Impl = type { i32, %struct.facet**, i64, %struct.facet**, i8** }
        %struct._Words = type { i8*, i64 }
        %"struct.__codecvt_abstract_base<char,char,__mbstate_t>" = type { %struct.facet }
        %"struct.basic_streambuf<char,std::char_traits<char> >" = type { i32 (...)**, i8*, i8*, i8*, i8*, i8*, i8*, %struct.locale }
        %struct.facet = type { i32 (...)**, i32 }
        %struct.ios_base = type { i32 (...)**, i64, i64, i32, i32, i32, %struct._Callback_list*, %struct._Words, [8 x %struct._Words], i32, %struct._Words*, %struct.locale }
        %struct.locale = type { %struct._Impl* }
        %"struct.ostreambuf_iterator<char,std::char_traits<char> >" = type { %"struct.basic_streambuf<char,std::char_traits<char> >"*, i1 }
@llvm.dbg.compile_unit1047 = external global %llvm.dbg.compile_unit.type          ; <%llvm.dbg.compile_unit.type*> [#uses=1]

define void @_ZNKSt7num_putIcSt19ostreambuf_iteratorIcSt11char_traitsIcEEE15_M_insert_floatIdEES3_S3_RSt8ios_baseccT_() {
entry:
        %tmp234 = icmp eq i8 0, 0               ; <i1> [#uses=1]
        br i1 %tmp234, label %cond_next243, label %cond_true235

cond_true235:           ; preds = %entry
        ret void

cond_next243:           ; preds = %entry
        %tmp428 = load i64* null                ; <i64> [#uses=1]
        %tmp428.upgrd.1 = trunc i64 %tmp428 to i32              ; <i32> [#uses=1]
        %tmp429 = alloca i8, i32 %tmp428.upgrd.1                ; <i8*> [#uses=0]
        call void @llvm.dbg.stoppoint( i32 1146, i32 0, {  }* bitcast (%llvm.dbg.compile_unit.type* @llvm.dbg.compile_unit1047 to {  }*) )
        unreachable
}

declare void @llvm.dbg.stoppoint(i32, i32, {  }*)

