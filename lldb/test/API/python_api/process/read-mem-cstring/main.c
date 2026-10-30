#include <stdlib.h>
int main ()
{
   const char *empty_string = "";
   const char *one_letter_string = "1";
   // This expects that lower 4k of memory will be mapped unreadable, which most
   // OSs do (to catch null pointer dereferences).
   const char *invalid_memory_string = (char*)0x100;
   // Not valid UTF-8: 0xc8 and 0xb9 are invalid as leading bytes.
   const char invalid_utf8_buf[] = {0xc8, 'd', 0xb9, '1', '\0'};
   const char *invalid_utf8_string = invalid_utf8_buf;

   return empty_string[0] + one_letter_string[0]; // breakpoint here
}
