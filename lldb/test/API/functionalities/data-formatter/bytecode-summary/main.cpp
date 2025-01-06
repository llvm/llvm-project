// A bare-bones llvm::Optional reimplementation.

template <typename T> struct MyOptionalStorage {
  MyOptionalStorage(T val) : value(val), hasVal(true) {}
  MyOptionalStorage() {}
  T value;
  bool hasVal = false;
};

template <typename T> struct MyOptional {
  MyOptionalStorage<T> Storage;
  MyOptional(T val) : Storage(val) {}
  MyOptional() {}
  T &operator*() { return Storage.value; }
};

void stop() {}

int main(int argc, char **argv) {
  MyOptional<int> x, y = 42;
  stop(); // break here
  return *y;
}

// Produced from the assembler in
// Shell/ScriptInterpreter/Python/Inputs/FormatterBytecode/formatter.py
__attribute__((used, section("__DATA_CONST,__lldbformatters"))) unsigned char
    _MyOptional_type_summary[] =
        "\x01"             // version
        "\xa4"             // record size
        "\x01"             // record size
        "\x10"             // type name size
        "^MyOptional<.+>$" // type name
        "\x00"             // flags
        "\x00"             // sig_summary
        "\x8d"             // program size
        "\x01"             // program size
        "\x1\x22\x7Storage#\x12\x60\x1,C\x10\x1\x5\x11\x2\x1\x22\x6hasVal#"
        "\x12\x60\x1,\x10\x1e\x2\x22\x1b<could not read MyOptional>\x10G#!\x60 "
        "\x0P\x10\x6\x22\x4None\x10\x36\x1#\x15\x60 "
        "\x0#\x16\x60\x5\x22\x5value#\x12\x60\x5#\x17\x60\x1,"
        "\x10\x6\x22\x4None\x10\x11\x1#\x0\x60\x1#R\x60\x10\x3# "
        "\x60\x10\x1\x2\x12\x12\x12\x12"; // summary function
