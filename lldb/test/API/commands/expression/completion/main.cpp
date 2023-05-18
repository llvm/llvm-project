#include <reserved.h>

namespace LongNamespaceName { class NestedClass { long m; }; }

// Defined in other.cpp, we only have a forward declaration here.
struct ForwardDecl;
extern ForwardDecl fwd_decl;

class LongClassName { long i ; };

class Expr {
public:
    int FooNoArgsBar() { return 1; }
    int FooWithArgsBar(int i) { return i; }
    int FooWithMultipleArgsBar(int i, int j) { return i + j; }
    int FooUnderscoreBar_() { return 4; }
    int FooNumbersBar1() { return 8; }
    int MemberVariableBar = 0;
    Expr &Self() { return *this; }
    static int StaticMemberMethodBar() { return 82; }
};

int main()
{
    LongClassName a;
    LongNamespaceName::NestedClass NestedFoo;
    long SomeLongVarNameWithCapitals = 44;
    int SomeIntVar = 33;
    Expr some_expr;
    some_expr.FooNoArgsBar();
    some_expr.FooWithArgsBar(1);
    some_expr.FooUnderscoreBar_();
    some_expr.FooNumbersBar1();
    Expr::StaticMemberMethodBar();
    ForwardDecl *fwd_decl_ptr = &fwd_decl;
    MyVec myVec;
    myVec.__func();
    myVec._Func();
    return 0; // Break here
}
