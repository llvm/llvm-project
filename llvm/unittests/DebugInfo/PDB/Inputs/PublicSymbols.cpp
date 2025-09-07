// clang-format off

// Compile with
// cl /Z7 /GR- /GS- PublicSymbols.cpp -c /Gy
// link .\PublicSymbols.obj /DEBUG /NODEFAULTLIB /out:PublicSymbols.exe /ENTRY:main /OPT:ICF
// llvm-pdbutil pdb2yaml --publics-stream PublicSymbols.pdb > PublicSymbols.yaml
// llvm-pdbutil yaml2pdb PublicSymbols.yaml
// 
// rm PublicSymbols.exe && rm PublicSymbols.obj && rm PublicSymbols.yaml

int foobar(int i){ return i + 1; }
// these should be merged with ICF
int dup1(int i){ return i + 2; }
int dup2(int i){ return i + 2; }
int dup3(int i){ return i + 2; }

class AClass {
public:
    void AMethod(int, char*) {}
    static bool Something(char c) {
        return c == ' ';
    }
};

struct Base {
    virtual ~Base() = default;
};
struct Derived : public Base {};
struct Derived2 : public Base {};
struct Derived3 : public Derived2, public Derived {};

int AGlobal;

void operator delete(void *,unsigned __int64) {}

int main() {
    foobar(1);
    dup1(1);
    dup2(1);
    dup3(1);
    AClass a;
    a.AMethod(1, nullptr);
    AClass::Something(' ');
    Derived3 d3;
    return AGlobal;
}
