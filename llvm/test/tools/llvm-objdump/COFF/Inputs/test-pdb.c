// Compile and generate YAML with:
//
// clang-cl test-pdb.c -fuse-ld=lld-link /Z7 /link /nodefaultlib /entry:main
// llvm-pdbutil pdb2yaml test-pdb.pdb --all > test-pdb.pdb.yaml
// obj2yaml test-pdb.exe -o test-pdb.exe.yaml
// rm test-pdb.exe && rm test-pdb.pdb

int square(int num) {
    return num * num;
}

int main() {
    return square(4);
}
