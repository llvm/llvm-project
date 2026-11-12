// This structure has a non-trivial copy constructor so
// it needs to be passed by reference.
struct PassByRef {
  PassByRef() = default;
  PassByRef(const PassByRef &p){x = p.x;};

  int x = 11223344;
};

PassByRef returnPassByRef() { return PassByRef(); }
int takePassByRef(PassByRef p) {
    return p.x;
}

int main() {
    PassByRef p = returnPassByRef();
    p.x = 42;
    return takePassByRef(p); // break here
}
