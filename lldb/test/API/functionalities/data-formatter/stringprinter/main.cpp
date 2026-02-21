#include <cstdio>
#include <cstring>

struct A {
  char data[4];
  char overflow[4];
};

#define MAKE_VARS3(c, v, s)                                                    \
  c v s char c##v##s##chararray[] = #c #v #s "char";                           \
  c v s char *c##v##s##charstar = c##v##s##chararray
#define MAKE_VARS2(c, v)                                                       \
  MAKE_VARS3(c, v, );                                                          \
  MAKE_VARS3(c, v, signed);                                                    \
  MAKE_VARS3(c, v, unsigned)
#define MAKE_VARS(c)                                                           \
  MAKE_VARS2(c, );                                                             \
  MAKE_VARS2(c, volatile)

MAKE_VARS();
MAKE_VARS(const);

template <typename T> struct S {
  int x = 0;
};
S<char[5]> Schar5;
S<char *> Scharstar;

int main(int argc, char const *argv[]) {
  const char manytrailingnuls[] = "F\0OO\0BA\0R\0\0\0\0";
  A a, b, c;
  // Deliberately write past the end of data to test that the formatter stops
  // at the end of array.
  memcpy(a.data, "FOOBAR", 7);
  memcpy(b.data, "FO\0BAR", 7);
  memcpy(c.data, "F\0O\0AR", 7);
  const char *charwithtabs = "Hello\t\tWorld\nI am here\t\tto say hello\n";
  const char *longconstcharstar =
      "I am a very long string; in fact I am longer than any reasonable length "
      "that a string should be; quite long indeed; oh my, so many words; so "
      "many letters; this is kind of like writing a poem; except in real life "
      "all that is happening"
      " is just me producing a very very long set of words; there is text "
      "here, text there, text everywhere; it fills me with glee to see so much "
      "text; all around me it's just letters, and symbols, and other pleasant "
      "drawings that cause me"
      " a large amount of joy upon visually seeing them with my eyes; well, "
      "this is now a lot of letters, but it is still not enough for the "
      "purpose of the test I want to test, so maybe I should copy and paste "
      "this a few times, you know.."
      " for science, or something"
      "I am a very long string; in fact I am longer than any reasonable length "
      "that a string should be; quite long indeed; oh my, so many words; so "
      "many letters; this is kind of like writing a poem; except in real life "
      "all that is happening"
      " is just me producing a very very long set of words; there is text "
      "here, text there, text everywhere; it fills me with glee to see so much "
      "text; all around me it's just letters, and symbols, and other pleasant "
      "drawings that cause me"
      " a large amount of joy upon visually seeing them with my eyes; well, "
      "this is now a lot of letters, but it is still not enough for the "
      "purpose of the test I want to test, so maybe I should copy and paste "
      "this a few times, you know.."
      " for science, or something"
      "I am a very long string; in fact I am longer than any reasonable length "
      "that a string should be; quite long indeed; oh my, so many words; so "
      "many letters; this is kind of like writing a poem; except in real life "
      "all that is happening"
      " is just me producing a very very long set of words; there is text "
      "here, text there, text everywhere; it fills me with glee to see so much "
      "text; all around me it's just letters, and symbols, and other pleasant "
      "drawings that cause me"
      " a large amount of joy upon visually seeing them with my eyes; well, "
      "this is now a lot of letters, but it is still not enough for the "
      "purpose of the test I want to test, so maybe I should copy and paste "
      "this a few times, you know.."
      " for science, or something";

  const char *basic = "Hello";
  const char *&ref = basic;
  const char *&&refref = "Hi";

  puts("Break here");

  return 0;
}
