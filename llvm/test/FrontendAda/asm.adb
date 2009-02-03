-- RUN: %llvmgcc -c %s
with System.Machine_Code;
procedure Asm is
begin
   System.Machine_Code.Asm ("");
end;
