-- RUN: %llvmgcc -c %s
procedure Array_Constructor is
   A : array (Integer range <>) of Boolean := (True, False);
begin
   null;
end;
