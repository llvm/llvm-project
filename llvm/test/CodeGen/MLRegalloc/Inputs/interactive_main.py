import log_reader
import interactive_host
import sys


def main(args):
  # this advisor just picks the first legal register to evict, which is
  # identifiable by the "mask" feature
  class Advisor:
    to_return = False

    def advice(self, tensor_values: list[log_reader.TensorValue]):
      for tv in tensor_values:
        if tv.spec().name != 'mask':
          continue
        for i, v in enumerate(tv):
          if v == 1:
            return i
      # i.e. invalid:
      return -1


  a = Advisor()
  interactive_host.run_interactive(args[0], a.advice, args[1:])


if __name__ == '__main__':
  main(sys.argv[1:])
