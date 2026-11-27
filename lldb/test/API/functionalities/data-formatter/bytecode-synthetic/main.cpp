struct Account {
  char *username;
  char *password;
};

void stop() {}

int main(int argc, char **argv) {
  struct Account acc;
  acc.username = "jake";
  acc.password = "popcorn";
  stop(); // break here
  return 0;
}

#ifdef __APPLE__
#define FORMATTER_SECTION "__DATA_CONST,__lldbformatters"
#else
#define FORMATTER_SECTION ".lldbformatters"
#endif

// clang-format off
__attribute__((used, section(FORMATTER_SECTION)))
unsigned char _Account_synthetic[] =
        "\x01"          // version
        "\x16"          // remaining record size
        "\x07"          // type name size
        "Account"       // type name
        "\x00"          // flags
        "\x06"          // sig_update
        "\x02"          // program size
        "\x20\x01"      // `return eReuse`
        "\x02"          // sig_get_num_children
        "\x02"          // program size
        "\x20\x01"      // `return 1`
        "\x04"          // sig_get_child_at_index
        "\x03"          // program size
        "\x23\x11\x60"; // `return self.valobj.GetChildAtIndex(idx)`
// clang-format on
