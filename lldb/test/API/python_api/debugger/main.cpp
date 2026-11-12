// This simple program is to test the lldb Python API SBDebugger.

int func(int val) {
    return val - 1;
}

int main (int argc, char const *argv[]) {
    return func(argc);
}
