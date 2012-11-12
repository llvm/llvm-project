namespace llvm {
namespace sys {
namespace locale {

int columnWidth(StringRef s) {
    return s.size();
}

bool isPrint(int c) {
    return ' ' <= c && c <= '~';
}

}
}
}