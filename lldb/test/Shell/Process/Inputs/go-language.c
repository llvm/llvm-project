#ifdef LLGO_LAYOUT
#define GO_TEXT "llgo"
#define GO_TEXT_LENGTH 4
#define GO_STRING_DATA data
#else
#define GO_TEXT "go"
#define GO_TEXT_LENGTH 2
#define GO_STRING_DATA str
#endif

typedef struct {
  const unsigned char *GO_STRING_DATA;
  long len;
} string;

typedef struct {
  long values[2];
} array2;

__attribute__((noinline)) long inspect(long value, string text, string *text_ptr, array2 array) { return value + array.values[0] + text.len + text_ptr->len; }

int main(void) {
  string text = {(const unsigned char *)GO_TEXT, GO_TEXT_LENGTH};
  string empty = {(const unsigned char *)"", 0};
  array2 array = {{11, 13}};
  long first = inspect(7, text, &text, array);
  long second = inspect(7, empty, &empty, array);
  return first + second == 0;
}
