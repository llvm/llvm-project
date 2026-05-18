// HTML-WHOLE-FILE: <td class='line-number'><a name='L[[@LINE+2]]' href='#L[[@LINE+2]]'><pre>[[@LINE+2]]</pre></a></td><td class='skipped-line'></td><td class='code'><pre>// before
// HTML-FILTER-NOT: <td class='line-number'><a name='L[[@LINE+1]]' href='#L[[@LINE+1]]'><pre>[[@LINE+1]]</pre></a></td><td class='skipped-line'></td><td class='code'><pre>// before
// before any coverage              // WHOLE-FILE: [[@LINE]]|                     |// before
                                    // FILTER-NOT: [[@LINE-1]]|                   |// before
// HTML: <td class='line-number'><a name='L[[@LINE+1]]' href='#L[[@LINE+1]]'><pre>[[@LINE+1]]</pre></a></td><td class='covered-line'><pre>161</pre></td><td class='code'><pre>int main() {
int main() {                              // TEXT: [[@LINE]]| [[#C161:min(C,161)]]|int main(
  int x = 0;                              // TEXT: [[@LINE]]|            [[#C161]]|  int x

  if (x) {                                // TEXT: [[@LINE]]|            [[#C161]]|  if (x)
    x = 0;                                // TEXT: [[@LINE]]|                    0|    x = 0
  } else {                                // TEXT: [[@LINE]]|            [[#C161]]|  } else
    x = 1;                                // TEXT: [[@LINE]]|            [[#C161]]|    x = 1
  }                                       // TEXT: [[@LINE]]|            [[#C161]]|  }

  for (int i = 0; i < 100; ++i) {         // TEXT: [[@LINE]]|            [[C16K2]]|  for (
    x = 1;                                // TEXT: [[@LINE]]|            [[C16K1]]|    x = 1
  }                                       // TEXT: [[@LINE]]|            [[C16K1]]|  }

  x = x < 10 ? x + 1 : x - 1;             // TEXT: [[@LINE]]|            [[#C161]]|  x =
  x = x > 10 ?                            // TEXT: [[@LINE]]|            [[#C161]]|  x =
        x - 1:                            // TEXT: [[@LINE]]|                    0|        x
        x + 1;                            // TEXT: [[@LINE]]|            [[#C161]]|        x

  return 0;                               // TEXT: [[@LINE]]|            [[#C161]]|  return
}                                         // TEXT: [[@LINE]]|            [[#C161]]|}
// after coverage                   // WHOLE-FILE: [[@LINE]]|                     |// after
                                    // FILTER-NOT: [[@LINE-1]]|                   |// after
// HTML-BINARY-NOT: <td class='covered-line'><pre>16
// HTML-WHOLE-FILE: <td class='line-number'><a name='L[[@LINE-3]]' href='#L[[@LINE-3]]'><pre>[[@LINE-3]]</pre></a></td><td class='skipped-line'></td><td class='code'><pre>// after
// HTML-FILTER-NOT: <td class='line-number'><a name='L[[@LINE-4]]' href='#L[[@LINE-4]]'><pre>[[@LINE-4]]</pre></a></td><td class='skipped-line'></td><td class='code'><pre>// after
