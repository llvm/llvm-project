-- RUN: %llvmgcc -c %s
procedure Placeholder is
   subtype Bounded is Integer range 1 .. 5;
   type Vector is array (Bounded range <>) of Integer;
   type Interval (Length : Bounded := 1) is record
      Points : Vector (1 .. Length);
   end record;
   An_Interval : Interval := (Length => 1, Points => (1 => 1));
   generic The_Interval : Interval; package R is end;
   package body R is end;
   package S is new R (An_Interval);
begin null; end;
