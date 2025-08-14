#define ATTRIB extern "C" __attribute__((noinline))
ATTRIB int function1_copy1(int a) {
  return ++a;
}

ATTRIB int function3_copy1(int a) {
    int b = a + 3;
    return b + 1;
}

ATTRIB int function2_copy1(int a) {
    return a - 22;
}

ATTRIB int function3_copy2(int a) {
    int b = a + 3;
    return b + 1;
}

ATTRIB int function2_copy2(int a) {
    int result = a - 22;
    return result;
}

struct logic_error {
    logic_error(const char* s) {}
};

struct length_error : public logic_error {
    __attribute__((noinline)) explicit length_error(const char* s) : logic_error(s) {}
};

int main() {
    int sum = 0;
    sum += function2_copy2(3);
    sum += function3_copy2(41);
    sum += function2_copy1(11);
    sum += function1_copy1(42);
    length_error e("test");
    return sum;
}
