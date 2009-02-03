-- RUN: %llvmgcc -c %s
procedure VCE is
  S : String (1 .. 2);
  B : Character := 'B';
begin
  S := 'A' & B;
end;
