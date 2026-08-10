void puts(const char *);

#define LLDBSUMMARY __attribute__((section("__TEXT,__lldbsummaries"), used))

struct Player {
  char *name;
  int number;
};

LLDBSUMMARY unsigned char _Player_type_summary[] =
    "\x01"                         // version
    "\x25"                         // record size
    "\x07"                         // type name size
    "Player\0"                     // type name
    "\x1c"                         // summary string size
    "${var.name} (${var.number})"; // summary string

struct Layer {
  char *name;
  int number;
};

LLDBSUMMARY unsigned char _padding[] = "\x00\x00";

// Near copy of the record for `Player`, using a regex type name (`^Layer`).
LLDBSUMMARY unsigned char _Layer_type_summary[] =
    "\x01"                         // version
    "\x25"                         // record size
    "\x07"                         // type name size
    "^Layer\0"                     // type name
    "\x1c"                         // summary string size
    "${var.name} (${var.number})"; // summary string

int main() {
  struct Player player;
  player.name = "Dirk";
  player.number = 41;
  struct Layer layer;
  layer.name = "crust";
  layer.number = 3;
  puts("break here");
  return 0;
}
