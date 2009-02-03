-- RUN: %llvmgcc -c %s
function Switch (N : Integer) return Integer is
begin
   case N is
      when Integer'First .. -1 =>
         return -1;
      when 0 =>
         return 0;
      when others =>
         return 1;
   end case;
end;
