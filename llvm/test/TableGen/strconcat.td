// RUN: tblgen %s | grep fufoo

class Y<string S> {
  string T = !strconcat(S, "foo");
}

def Z : Y<"fu">;
