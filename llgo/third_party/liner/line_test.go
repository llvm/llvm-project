package liner

import (
	"bytes"
	"fmt"
	"strings"
	"testing"
)

func TestAppend(t *testing.T) {
	var s State
	s.AppendHistory("foo")
	s.AppendHistory("bar")

	var out bytes.Buffer
	num, err := s.WriteHistory(&out)
	if err != nil {
		t.Fatal("Unexpected error writing history", err)
	}
	if num != 2 {
		t.Fatalf("Expected 2 history entries, got %d", num)
	}

	s.AppendHistory("baz")
	num, err = s.WriteHistory(&out)
	if err != nil {
		t.Fatal("Unexpected error writing history", err)
	}
	if num != 3 {
		t.Fatalf("Expected 3 history entries, got %d", num)
	}

	s.AppendHistory("baz")
	num, err = s.WriteHistory(&out)
	if err != nil {
		t.Fatal("Unexpected error writing history", err)
	}
	if num != 3 {
		t.Fatalf("Expected 3 history entries after duplicate append, got %d", num)
	}

	s.AppendHistory("baz")

}

func TestHistory(t *testing.T) {
	input := `foo
bar
baz
quux
dingle`

	var s State
	num, err := s.ReadHistory(strings.NewReader(input))
	if err != nil {
		t.Fatal("Unexpected error reading history", err)
	}
	if num != 5 {
		t.Fatal("Wrong number of history entries read")
	}

	var out bytes.Buffer
	num, err = s.WriteHistory(&out)
	if err != nil {
		t.Fatal("Unexpected error writing history", err)
	}
	if num != 5 {
		t.Fatal("Wrong number of history entries written")
	}
	if strings.TrimSpace(out.String()) != input {
		t.Fatal("Round-trip failure")
	}

	// Test reading with a trailing newline present
	var s2 State
	num, err = s2.ReadHistory(&out)
	if err != nil {
		t.Fatal("Unexpected error reading history the 2nd time", err)
	}
	if num != 5 {
		t.Fatal("Wrong number of history entries read the 2nd time")
	}

	num, err = s.ReadHistory(strings.NewReader(input + "\n\xff"))
	if err == nil {
		t.Fatal("Unexpected success reading corrupted history", err)
	}
	if num != 5 {
		t.Fatal("Wrong number of history entries read the 3rd time")
	}
}

func TestColumns(t *testing.T) {
	list := []string{"foo", "food", "This entry is quite a bit longer than the typical entry"}

	output := []struct {
		width, columns, rows, maxWidth int
	}{
		{80, 1, 3, len(list[2]) + 1},
		{120, 2, 2, len(list[2]) + 1},
		{800, 14, 1, 0},
		{8, 1, 3, 7},
	}

	for i, o := range output {
		col, row, max := calculateColumns(o.width, list)
		if col != o.columns {
			t.Fatalf("Wrong number of columns, %d != %d, in TestColumns %d\n", col, o.columns, i)
		}
		if row != o.rows {
			t.Fatalf("Wrong number of rows, %d != %d, in TestColumns %d\n", row, o.rows, i)
		}
		if max != o.maxWidth {
			t.Fatalf("Wrong column width, %d != %d, in TestColumns %d\n", max, o.maxWidth, i)
		}
	}
}

// This example demonstrates a way to retrieve the current
// history buffer without using a file.
func ExampleState_WriteHistory() {
	var s State
	s.AppendHistory("foo")
	s.AppendHistory("bar")

	buf := new(bytes.Buffer)
	_, err := s.WriteHistory(buf)
	if err == nil {
		history := strings.Split(strings.TrimSpace(buf.String()), "\n")
		for i, line := range history {
			fmt.Println("History entry", i, ":", line)
		}
	}
	// Output:
	// History entry 0 : foo
	// History entry 1 : bar
}
