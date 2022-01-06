#include <string>
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

template<typename T>
struct S {
  int x = 0;
};
S<char[5]> Schar5;
S<char *> Scharstar;

int main (int argc, char const *argv[])
{
    A a, b, c;
    // Deliberately write past the end of data to test that the formatter stops
    // at the end of array.
    memcpy(a.data, "FOOBAR", 7);
    memcpy(b.data, "FO\0BAR", 7);
    memcpy(c.data, "F\0O\0AR", 7);
    std::string stdstring("Hello\t\tWorld\nI am here\t\tto say hello\n"); //%self.addTearDownHook(lambda x: x.runCmd("setting set escape-non-printables true"))
    const char *charwithtabs = stdstring.c_str();
    std::string longstring(
"I am a very long string; in fact I am longer than any reasonable length that a string should be; quite long indeed; oh my, so many words; so many letters; this is kind of like writing a poem; except in real life all that is happening"
" is just me producing a very very long set of words; there is text here, text there, text everywhere; it fills me with glee to see so much text; all around me it's just letters, and symbols, and other pleasant drawings that cause me"
" a large amount of joy upon visually seeing them with my eyes; well, this is now a lot of letters, but it is still not enough for the purpose of the test I want to test, so maybe I should copy and paste this a few times, you know.."
" for science, or something"
      "I am a very long string; in fact I am longer than any reasonable length that a string should be; quite long indeed; oh my, so many words; so many letters; this is kind of like writing a poem; except in real life all that is happening"
      " is just me producing a very very long set of words; there is text here, text there, text everywhere; it fills me with glee to see so much text; all around me it's just letters, and symbols, and other pleasant drawings that cause me"
      " a large amount of joy upon visually seeing them with my eyes; well, this is now a lot of letters, but it is still not enough for the purpose of the test I want to test, so maybe I should copy and paste this a few times, you know.."
      " for science, or something"
            "I am a very long string; in fact I am longer than any reasonable length that a string should be; quite long indeed; oh my, so many words; so many letters; this is kind of like writing a poem; except in real life all that is happening"
            " is just me producing a very very long set of words; there is text here, text there, text everywhere; it fills me with glee to see so much text; all around me it's just letters, and symbols, and other pleasant drawings that cause me"
            " a large amount of joy upon visually seeing them with my eyes; well, this is now a lot of letters, but it is still not enough for the purpose of the test I want to test, so maybe I should copy and paste this a few times, you know.."
            " for science, or something"
      );
    const char* longconstcharstar = longstring.c_str();
    return 0;     //% if self.TraceOn(): self.runCmd('frame variable')
    //%
    //% self.expect_var_path('stdstring', summary='"Hello\\t\\tWorld\\nI am here\\t\\tto say hello\\n"')
    //% self.expect_var_path('charwithtabs', summary='"Hello\\t\\tWorld\\nI am here\\t\\tto say hello\\n"')
    //% self.expect_var_path("a.data", summary='"FOOB"')
    //% self.expect_var_path("b.data", summary=r'"FO\0B"')
    //% self.expect_var_path("c.data", summary=r'"F\0O"')
    //%
    //% for c in ["", "const"]:
    //%   for v in ["", "volatile"]:
    //%     for s in ["", "unsigned"]:
    //%       summary = '"'+c+v+s+'char"'
    //%       self.expect_var_path(c+v+s+"chararray", summary=summary)
    //% # These should be printed normally
    //%       self.expect_var_path(c+v+s+"charstar", summary=summary)
    //% Schar5 = self.expect_var_path("Schar5",
    //%     children=[ValueCheck(name="x", value="0")])
    //% self.assertIsNone(Schar5.GetSummary())
    //% Scharstar = self.expect_var_path("Scharstar",
    //%     children=[ValueCheck(name="x", value="0")])
    //% self.assertIsNone(Scharstar.GetSummary())
    //%
    //% self.runCmd("setting set escape-non-printables false")
    //% self.expect_var_path('stdstring', summary='"Hello\t\tWorld\nI am here\t\tto say hello\n"')
    //% self.expect_var_path('charwithtabs', summary='"Hello\t\tWorld\nI am here\t\tto say hello\n"')
    //% self.assertTrue(self.frame().FindVariable('longstring').GetSummary().endswith('"...'))
    //% self.assertTrue(self.frame().FindVariable('longconstcharstar').GetSummary().endswith('"...'))
    // FIXME: make "b.data" and "c.data" work sanely
}

